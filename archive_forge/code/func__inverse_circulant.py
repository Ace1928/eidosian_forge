from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import linear_operator_block_diag
from tensorflow.python.ops.linalg import linear_operator_block_lower_triangular
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_householder
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_inversion
from tensorflow.python.ops.linalg import linear_operator_kronecker
@linear_operator_algebra.RegisterInverse(linear_operator_circulant._BaseLinearOperatorCirculant)
def _inverse_circulant(circulant_operator):
    return circulant_operator.__class__(spectrum=1.0 / circulant_operator.spectrum, is_non_singular=circulant_operator.is_non_singular, is_self_adjoint=circulant_operator.is_self_adjoint, is_positive_definite=circulant_operator.is_positive_definite, is_square=True, input_output_dtype=circulant_operator.dtype)