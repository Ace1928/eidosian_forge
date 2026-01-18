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
class _AddAndReturnDiag(_Adder):
    """Handles additions resulting in a Diag operator."""

    def can_add(self, op1, op2):
        types = {_type(op1), _type(op2)}
        return not types.difference(_DIAG_LIKE)

    def _add(self, op1, op2, operator_name, hints):
        return linear_operator_diag.LinearOperatorDiag(diag=op1.diag_part() + op2.diag_part(), is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)