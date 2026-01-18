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
class _AddAndReturnTriL(_Adder):
    """Handles additions resulting in a TriL operator."""

    def can_add(self, op1, op2):
        types = {_type(op1), _type(op2)}
        return not types.difference(_DIAG_LIKE.union({_TRIL}))

    def _add(self, op1, op2, operator_name, hints):
        if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
            op_add_to_tensor, op_other = (op1, op2)
        else:
            op_add_to_tensor, op_other = (op2, op1)
        return linear_operator_lower_triangular.LinearOperatorLowerTriangular(tril=op_add_to_tensor.add_to_tensor(op_other.to_dense()), is_non_singular=hints.is_non_singular, is_self_adjoint=hints.is_self_adjoint, is_positive_definite=hints.is_positive_definite, name=operator_name)