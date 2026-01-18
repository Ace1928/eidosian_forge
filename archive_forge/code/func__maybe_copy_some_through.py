from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()
    nest.assert_same_structure(zero_output, new_output)
    nest.assert_same_structure(state, new_state)
    flat_new_state = nest.flatten(new_state)
    flat_new_output = nest.flatten(new_output)
    return cond.cond(time < min_sequence_length, lambda: flat_new_output + flat_new_state, lambda: _copy_some_through(flat_new_output, flat_new_state))