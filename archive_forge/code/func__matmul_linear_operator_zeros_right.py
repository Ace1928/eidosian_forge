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
@linear_operator_algebra.RegisterMatmul(linear_operator.LinearOperator, linear_operator_zeros.LinearOperatorZeros)
def _matmul_linear_operator_zeros_right(linop, zeros):
    if not zeros.is_square or not linop.is_square:
        raise ValueError('Matmul with non-square `LinearOperator`s or non-square `LinearOperatorZeros` not supported at this time.')
    return zeros