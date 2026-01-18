from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def _get_dependent_variables(input_ops, output_ops):
    """Finds variables involved in the subgraph between input_ops and output_ops.

  Args:
    input_ops: Flattened list of input ops
    output_ops: Flattened list of output ops

  Returns:
    A list of variables
  """
    output_ops = nest.map_structure(gen_array_ops.identity, output_ops)
    inbetween_ops = op_selector.get_backward_walk_ops(seed_ops=output_ops, stop_at_ts=input_ops, inclusive=False, only_differentiable=True)
    var_ops = (op for op in inbetween_ops if op.type in VAR_OP_TYPES)
    var_names = (op.name for op in var_ops)
    tf_vars = (get_variable_by_name(var_name) for var_name in var_names)
    tf_vars = [v for v in tf_vars if v is not None]
    return tf_vars