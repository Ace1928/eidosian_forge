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
def _is_llt_product(linop):
    """Determines if linop = L @ L.H for L = LinearOperatorLowerTriangular."""
    if len(linop.operators) != 2:
        return False
    if not linear_operator_util.is_aat_form(linop.operators):
        return False
    return isinstance(linop.operators[0], LinearOperatorLowerTriangular)