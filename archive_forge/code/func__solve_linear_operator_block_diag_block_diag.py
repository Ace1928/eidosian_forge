from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import registrations_util
@linear_operator_algebra.RegisterSolve(linear_operator_block_diag.LinearOperatorBlockDiag, linear_operator_block_diag.LinearOperatorBlockDiag)
def _solve_linear_operator_block_diag_block_diag(linop_a, linop_b):
    return linear_operator_block_diag.LinearOperatorBlockDiag(operators=[o1.solve(o2) for o1, o2 in zip(linop_a.operators, linop_b.operators)], is_non_singular=registrations_util.combined_non_singular_hint(linop_a, linop_b), is_self_adjoint=None, is_positive_definite=None, is_square=True)