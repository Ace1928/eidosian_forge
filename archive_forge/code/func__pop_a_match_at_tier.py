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
def _pop_a_match_at_tier(op1, operator_list, tier):
    for i in range(1, len(operator_list) + 1):
        op2 = operator_list[-i]
        for adder in tier:
            if adder.can_add(op1, op2):
                return (operator_list.pop(-i), adder)
    return (None, None)