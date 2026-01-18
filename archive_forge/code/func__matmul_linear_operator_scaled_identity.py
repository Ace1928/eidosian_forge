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
@linear_operator_algebra.RegisterMatmul(linear_operator_identity.LinearOperatorScaledIdentity, linear_operator_identity.LinearOperatorScaledIdentity)
def _matmul_linear_operator_scaled_identity(linop_a, linop_b):
    """Matmul of two ScaledIdentity `LinearOperators`."""
    return linear_operator_identity.LinearOperatorScaledIdentity(num_rows=linop_a.domain_dimension_tensor(), multiplier=linop_a.multiplier * linop_b.multiplier, is_non_singular=registrations_util.combined_non_singular_hint(linop_a, linop_b), is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(linop_a, linop_b), is_positive_definite=registrations_util.combined_commuting_positive_definite_hint(linop_a, linop_b), is_square=True)