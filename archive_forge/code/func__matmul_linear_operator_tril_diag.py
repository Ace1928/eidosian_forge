from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_composition
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_zeros
from tensorflow.python.ops.linalg import registrations_util
@linear_operator_algebra.RegisterMatmul(linear_operator_lower_triangular.LinearOperatorLowerTriangular, linear_operator_diag.LinearOperatorDiag)
def _matmul_linear_operator_tril_diag(linop_triangular, linop_diag):
    return linear_operator_lower_triangular.LinearOperatorLowerTriangular(tril=linop_triangular.to_dense() * linop_diag.diag, is_non_singular=registrations_util.combined_non_singular_hint(linop_diag, linop_triangular), is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(linop_diag, linop_triangular), is_positive_definite=None, is_square=True)