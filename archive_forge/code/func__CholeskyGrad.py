from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('Cholesky')
def _CholeskyGrad(op, grad):
    """Gradient for Cholesky."""
    l = op.outputs[0]
    num_rows = array_ops.shape(l)[-1]
    batch_shape = array_ops.shape(l)[:-2]
    l_inverse = linalg_ops.matrix_triangular_solve(l, linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=l.dtype))
    middle = math_ops.matmul(l, grad, adjoint_a=True)
    middle = array_ops.matrix_set_diag(middle, 0.5 * array_ops.matrix_diag_part(middle))
    middle = array_ops.matrix_band_part(middle, -1, 0)
    grad_a = math_ops.matmul(math_ops.matmul(l_inverse, middle, adjoint_a=True), l_inverse)
    grad_a += _linalg.adjoint(grad_a)
    return grad_a * 0.5