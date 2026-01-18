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
@linear_operator_algebra.RegisterCholesky(linear_operator_diag.LinearOperatorDiag)
def _cholesky_diag(diag_operator):
    return linear_operator_diag.LinearOperatorDiag(math_ops.sqrt(diag_operator.diag), is_non_singular=True, is_self_adjoint=True, is_positive_definite=True, is_square=True)