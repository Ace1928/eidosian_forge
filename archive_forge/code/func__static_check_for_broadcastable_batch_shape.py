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
def _static_check_for_broadcastable_batch_shape(operators):
    """ValueError if operators determined to have non-broadcastable shapes."""
    if len(operators) < 2:
        return
    batch_shape = operators[0].batch_shape
    for op in operators[1:]:
        batch_shape = array_ops.broadcast_static_shape(batch_shape, op.batch_shape)