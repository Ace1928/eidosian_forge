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
def _time_step(time, output_ta_t, state):
    """Take a time step of the dynamic RNN.

    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.

    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """
    if in_graph_mode:
        input_t = tuple((ta.read(time) for ta in input_ta))
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])
    else:
        input_t = tuple((ta[time.numpy()] for ta in input_ta))
    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    call_cell = lambda: cell(input_t, state)
    if sequence_length is not None:
        output, new_state = _rnn_step(time=time, sequence_length=sequence_length, min_sequence_length=min_sequence_length, max_sequence_length=max_sequence_length, zero_output=zero_output, state=state, call_cell=call_cell, state_size=state_size, skip_conditionals=True)
    else:
        output, new_state = call_cell()
    output = nest.flatten(output)
    if in_graph_mode:
        output_ta_t = tuple((ta.write(time, out) for ta, out in zip(output_ta_t, output)))
    else:
        for ta, out in zip(output_ta_t, output):
            ta[time.numpy()] = out
    return (time + 1, output_ta_t, new_state)