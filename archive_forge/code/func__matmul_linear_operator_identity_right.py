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
@linear_operator_algebra.RegisterMatmul(linear_operator.LinearOperator, linear_operator_identity.LinearOperatorIdentity)
def _matmul_linear_operator_identity_right(linop, identity):
    del identity
    return linop