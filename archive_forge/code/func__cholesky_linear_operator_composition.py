from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_kronecker
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_util
@linear_operator_algebra.RegisterCholesky(linear_operator_composition.LinearOperatorComposition)
def _cholesky_linear_operator_composition(linop):
    """Computes Cholesky(LinearOperatorComposition)."""
    if not _is_llt_product(linop):
        return LinearOperatorLowerTriangular(linalg_ops.cholesky(linop.to_dense()), is_non_singular=True, is_self_adjoint=False, is_square=True)
    left_op = linop.operators[0]
    if left_op.is_positive_definite:
        return left_op
    diag_sign = array_ops.expand_dims(math_ops.sign(left_op.diag_part()), axis=-2)
    return LinearOperatorLowerTriangular(tril=left_op.tril / diag_sign, is_non_singular=left_op.is_non_singular, is_self_adjoint=left_op.is_self_adjoint, is_positive_definite=True if left_op.is_positive_definite else None, is_square=True)