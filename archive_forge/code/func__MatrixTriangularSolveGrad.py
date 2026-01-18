from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('MatrixTriangularSolve')
def _MatrixTriangularSolveGrad(op, grad):
    """Gradient for MatrixTriangularSolve."""
    a = op.inputs[0]
    b = op.inputs[1]
    adjoint_a = op.get_attr('adjoint')
    lower_a = op.get_attr('lower')
    c = op.outputs[0]
    grad_b = linalg_ops.matrix_triangular_solve(a, grad, lower=lower_a, adjoint=not adjoint_a)
    if adjoint_a:
        grad_a = -math_ops.matmul(c, grad_b, adjoint_b=True)
    else:
        grad_a = -math_ops.matmul(grad_b, c, adjoint_b=True)
    if lower_a:
        grad_a = array_ops.matrix_band_part(grad_a, -1, 0)
    else:
        grad_a = array_ops.matrix_band_part(grad_a, 0, -1)
    if a.shape.is_fully_defined() and b.shape.is_fully_defined() and (a.shape[:-2] == b.shape[:-2]):
        return (grad_a, grad_b)
    a_shape = array_ops.shape(a)
    b_shape = array_ops.shape(b)
    ra, rb = array_ops.broadcast_gradient_args(a_shape[:-2], b_shape[:-2])
    grad_a = array_ops.reshape(math_ops.reduce_sum(grad_a, axis=ra), a_shape)
    grad_b = array_ops.reshape(math_ops.reduce_sum(grad_b, axis=rb), b_shape)
    return (grad_a, grad_b)