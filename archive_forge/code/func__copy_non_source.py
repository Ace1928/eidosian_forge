import collections
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import op_selector
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _copy_non_source(op, graph, op_map, base_graph):
    """Copy an op directly to a given graph.

  Generally `op`'s inputs should already have been copied. If this is not the
  case, for example with v1 while_loops, then `_copy_non_source` inserts
  placeholders for the unavailable Tensors and returns a list of required
  mutations.

  Args:
    op: The op to be copied.
    graph: The destination graph.
    op_map: A dict mapping ops and tensors in the old graph to the new one.
    base_graph: The graph we're copying from, for any necessary functions.
  Returns:
    A tuple of (required_inputs, required_control_inputs):
      required_inputs:
        A list of `_InputMutation` tuples containing inputs to `copied_op` which
        must be updated once `old_graph_tensor` has been copied.
      required_control_inputs:
        A list of `_ControlMutation` tuples containing control inputs to
        `copied_op` which must be added once `old_graph_op` has been copied.
  """
    input_mutations = []
    control_mutations = []
    copied_inputs = []
    for input_index, original_input in enumerate(op.inputs):
        copied_input = op_map.get(original_input, None)
        if copied_input is None:
            copied_input = array_ops.placeholder(name='unused_control_flow_input', shape=original_input.shape, dtype=original_input.dtype)
            input_mutations.append(_InputMutation(copied_op=None, input_index=input_index, old_graph_tensor=original_input))
        copied_inputs.append(copied_input)
    copied_control_inputs = []
    for original_control_input in op.control_inputs:
        copied_control_input = op_map.get(original_control_input, None)
        if copied_control_input is None:
            control_mutations.append(_ControlMutation(copied_op=None, old_graph_op=original_control_input))
        else:
            copied_control_inputs.append(copied_control_input)
    with ops.control_dependencies(copied_control_inputs), ops.device(op.device):
        f = base_graph._functions.get(op.type, None)
        if f is not None and compat.as_str(f.name) not in graph._functions:
            f.add_to_graph(graph)
        copied_op = graph.create_op(op_type=op.type, inputs=copied_inputs, dtypes=[x.dtype for x in op.outputs], attrs={key: value for key, value in op.node_def.attr.items() if not key.startswith('_class') and (not key.startswith('_tpu_replicate'))}, name=op.name)
    op_map[op] = copied_op
    for i, o in enumerate(op.outputs):
        op_map[o] = copied_op.outputs[i]
    return ([mutation._replace(copied_op=copied_op) for mutation in input_mutations], [mutation._replace(copied_op=copied_op) for mutation in control_mutations])