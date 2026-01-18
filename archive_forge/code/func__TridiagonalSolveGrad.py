from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
@ops.RegisterGradient('TridiagonalSolve')
def _TridiagonalSolveGrad(op, grad):
    """Gradient for TridiagonalSolveGrad."""
    diags = op.inputs[0]
    x = op.outputs[0]
    partial_pivoting = op.get_attr('partial_pivoting')
    perturb_singular = op.get_attr('perturb_singular')
    diags_transposed = _TransposeTridiagonalMatrix(diags)
    grad_rhs = linalg_ops.tridiagonal_solve(diags_transposed, grad, partial_pivoting=partial_pivoting, perturb_singular=perturb_singular)
    grad_diags = -_MatmulExtractingThreeDiagonals(grad_rhs, x)
    return (grad_diags, grad_rhs)