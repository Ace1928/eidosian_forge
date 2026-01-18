from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseToDense')
def _SparseToDenseGrad(op, grad):
    sparse_indices, output_shape, _, _ = op.inputs
    sparse_values_grad = array_ops.gather_nd(grad, sparse_indices)
    default_value_grad = math_ops.reduce_sum(grad) - math_ops.reduce_sum(sparse_values_grad)
    return [array_ops.zeros_like(sparse_indices), array_ops.zeros_like(output_shape), sparse_values_grad, default_value_grad]