from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _Underdetermined(op, grad):
    """Gradients for the underdetermined case of MatrixSolveLs.

    This is the backprop for the solution to the normal equations of the second
    kind:
      X = F(A, B) = A * (A*A^T + lambda*I)^{-1} * B
    that (for lambda=0) solve the least squares problem
      min ||X||_F subject to A*X = B.
    """
    a = op.inputs[0]
    b = op.inputs[1]
    l2_regularizer = math_ops.cast(op.inputs[2], a.dtype.base_dtype)
    chol = linalg_ops._RegularizedGramianCholesky(a, l2_regularizer=l2_regularizer, first_kind=False)
    grad_b = linalg_ops.cholesky_solve(chol, math_ops.matmul(a, grad))
    tmp = linalg_ops.cholesky_solve(chol, b)
    a1 = math_ops.matmul(tmp, a, adjoint_a=True)
    a1 = -math_ops.matmul(grad_b, a1)
    a2 = grad - math_ops.matmul(a, grad_b, adjoint_a=True)
    a2 = math_ops.matmul(tmp, a2, adjoint_b=True)
    grad_a = a1 + a2
    return (grad_a, grad_b, None)