import abc
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular
class _Hints:
    """Holds 'is_X' flags that every LinearOperator is initialized with."""

    def __init__(self, is_non_singular=None, is_positive_definite=None, is_self_adjoint=None):
        self.is_non_singular = is_non_singular
        self.is_positive_definite = is_positive_definite
        self.is_self_adjoint = is_self_adjoint