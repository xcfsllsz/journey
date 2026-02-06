## Float32 浮点数表示

**公式**：$\text{数值} = (-1)^S \times (1 + M/2^{23}) \times 2^{(E - 127)}$

### 位结构详解

**符号位（S）**：1 位
- 决定数值的正负号
- $S=0$ 表示正数，$S=1$ 表示负数

**指数位（E）**：8 位
- 二进制取值范围：0～255
- 实际指数 = E - 127，范围为 -127～+128
- **特殊值处理**：
  - $E=0$：表示非规格化数，用于表示接近 0 的极小值
  - $E=255$：表示无穷大（Inf）或非数值（NaN）

**尾数位（M）**：23 位
- 表示小数部分的 23 位二进制数
- **隐含的"1"**：IEEE 754 标准的优化设计
  - 因为二进制有效数字的最高位必定是 1
  - 实际尾数为 $1.M$，而非仅 $M$
  
**例子**：尾数位为 $101_2$，则 $M = 1 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} = 0.625$，实际尾数为 $1.625$
## 张量转置后为什么 view() 会报错？

**问题根源**：张量内存布局与逻辑形状不匹配

**详细分析**：

1. **原始张量 y**（2 行 3 列）：
   - 内存中连续存储：`1 → 2 → 3 → 4 → 5 → 6`（行优先存储）

2. **转置后张量 y_t**（3 行 2 列）：
   - PyTorch 采用 **惰性转置**（lazy transpose）策略
   - 不重新排列内存数据，而是通过修改"步长（stride）"来模拟转置
   - 元素访问顺序变为：`1 → 4 → 2 → 5 → 3 → 6`
   - 但物理内存依然是：`1 → 2 → 3 → 4 → 5 → 6`

3. **view() 操作失败的原因**：
   - `view()` 要求张量在内存中必须是**连续的块**
   - 转置后的张量内存分布不连续，无法直接按新形状切割内存
   - 因此抛出错误：`RuntimeError: view size is not compatible with input tensor's size`

**解决方案**：调用 `.contiguous()` 重新整理内存后再使用 `view()`
## 矩阵乘法计算复杂度

- **主要计算量**：矩阵乘法占主导 $(2mnp)$ FLOP
- **性能影响因素**：
  - 硬件性能：H100 >> A100
  - 数据类型：bfloat16 >> float32
- **模型浮点运算利用率（MFU）**：
  $$\text{MFU} = \frac{\text{实际 FLOP/s}}{\text{承诺 FLOP/s}}$$
## 向量点积求导：数学推导

### 1. 变量定义

设输入向量 $\mathbf{x}$ 和权重向量 $\mathbf{w}$ 均为 $n$ 维列向量：
- $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top$
- $\mathbf{w} = [w_1, w_2, \dots, w_n]^\top$
- $b$ 为偏置常数

定义标量损失函数：
$$\text{loss} = f(\mathbf{w}) = \frac{1}{2}(\mathbf{x}^\top \mathbf{w} - b)^2$$

### 2. 链式法则分解

为了求 $\frac{\partial \text{loss}}{\partial \mathbf{w}}$，引入中间变量 $u$：
- **内层函数**：$u = \mathbf{x}^\top \mathbf{w} - b = \sum_{i=1}^n x_i w_i - b$
- **外层函数**：$\text{loss} = \frac{1}{2} u^2$

根据链式法则：
$$\frac{\partial \text{loss}}{\partial \mathbf{w}} = \frac{\partial \text{loss}}{\partial u} \cdot \frac{\partial u}{\partial \mathbf{w}}$$

### 3. 分步求导

**第一步**：外层标量对标量求导
$$\frac{\partial \text{loss}}{\partial u} = \frac{\partial}{\partial u}\left(\frac{1}{2}u^2\right) = u$$

**第二步**：内层标量对向量求导

对 $\mathbf{w}$ 的第 $j$ 个分量求偏导：
$$\frac{\partial u}{\partial w_j} = \frac{\partial}{\partial w_j}\left(\sum_{i=1}^n x_i w_i - b\right) = x_j$$

组合所有分量：
$$\frac{\partial u}{\partial \mathbf{w}} = [x_1, x_2, \dots, x_n]^\top = \mathbf{x}$$

### 4. 最终结果

$$\frac{\partial \text{loss}}{\partial \mathbf{w}} = u \cdot \mathbf{x} = (\mathbf{x}^\top \mathbf{w} - b)\mathbf{x}$$

### 5. 数值验证

给定：
- $\mathbf{x} = [1, 2, 3]^\top$
- $\mathbf{w} = [1, 1, 1]^\top$
- $b = 5$

计算步骤：
- $u = (1 \times 1 + 2 \times 1 + 3 \times 1) - 5 = 6 - 5 = 1$
- $\nabla_{\mathbf{w}} \text{loss} = 1 \cdot [1, 2, 3]^\top = [1, 2, 3]^\top$ ✓
## 前向传播与反向传播

### 前向传播（Forward Pass）

$$\mathbf{H}_{\text{out}} = \mathbf{H}_{\text{in}} \mathbf{W}$$

### 反向传播（Backward Pass）

$$\nabla_{\mathbf{W}} L = \mathbf{H}_{\text{in}}^\top \nabla_{\mathbf{H}_{\text{out}}} L$$

### 计算复杂度

无论前向传播还是反向传播，计算复杂度均为 $O(2 \times B \times D \times K)$

其中 $B$ 为批次大小，$D$ 为隐藏维度，$K$ 为输出维度。

### 反向传播全过程示例

对于链路 $x \to w_1 \to h_1 \to w_2 \to h_2 \to \text{loss}$：

1. **初始化**：$\frac{\partial \text{loss}}{\partial h_2} = \nabla_{h_2} L$

2. **第二层梯度**：$\frac{\partial L}{\partial w_2} = h_1^\top \cdot \nabla_{h_2} L$

3. **第一层梯度传递**：$\frac{\partial L}{\partial h_1} = \nabla_{h_2} L \cdot w_2^\top$

4. **第一层权重梯度**：$\frac{\partial L}{\partial w_1} = x^\top \cdot \nabla_{h_1} L$
