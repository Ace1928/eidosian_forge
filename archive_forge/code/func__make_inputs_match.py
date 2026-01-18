import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _make_inputs_match(branch_graphs, branch_inputs):
    """Modifies branch_graphs so they have the same input signature.

  This method reorders and/or adds parameters to each graph in branch_graphs so
  they have the same input signature, and updates the 'inputs' and 'captured'
  fields of each graph accordingly. It uses the input tensors from the outer
  graph to avoid duplicating shared arguments.

  Args:
    branch_graphs: a `list` of `FuncGraph`
    branch_inputs: a `list` of `list`s of `Tensor`s in the outer graph. The
      inputs for the corresponding graph in `branch_graphs`.

  Returns:
    A new list of Tensors from the outer graph that are the new inputs for each
    branch_graph. This is a deduped version of `sum(branch_inputs)`.
  """
    assert len(branch_graphs) == len(branch_inputs)
    added_inputs = set()
    new_inputs = []
    for branch_in in branch_inputs:
        for tensor in branch_in:
            tensor_id = ops.tensor_id(tensor)
            if tensor_id not in added_inputs:
                added_inputs.add(tensor_id)
                new_inputs.append(tensor)
    for branch_graph, branch_in in zip(branch_graphs, branch_inputs):
        input_ids = [ops.tensor_id(t) for t in branch_in]
        branch_input_to_param = dict(zip(input_ids, branch_graph.inputs))
        input_list = []
        for in_t in new_inputs:
            param = branch_input_to_param.get(ops.tensor_id(in_t))
            if param is None:
                param = _create_dummy_input(branch_graph, in_t)
            input_list.append(param)
        branch_graph.inputs = input_list
        branch_graph.function_captures.reset_captures(new_inputs, branch_graph.inputs)
    return new_inputs