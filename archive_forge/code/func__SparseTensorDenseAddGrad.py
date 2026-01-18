from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('SparseTensorDenseAdd')
def _SparseTensorDenseAddGrad(op, out_grad):
    sp_indices = op.inputs[0]
    return (None, array_ops.gather_nd(out_grad, sp_indices), None, out_grad)