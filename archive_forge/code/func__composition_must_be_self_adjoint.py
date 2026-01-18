from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export
def _composition_must_be_self_adjoint(operators):
    """Runs some checks to see if composition operators must be SA.

  Args:
    operators: List of LinearOperators.

  Returns:
    True if the composition must be SA. False if it is not SA OR if we did not
      determine whether the composition is SA.
  """
    if len(operators) == 1 and operators[0].is_self_adjoint:
        return True
    if linear_operator_util.is_aat_form(operators):
        return True
    return False