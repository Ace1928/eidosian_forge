import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils
def _resource_capture_helper(self, tensor):
    """Returns the captured resource tensor.

    Resource-type tensors are not accumulated. If a resource tensor exists in
    the loop body it must either be a loop input or an output of a nested While
    op inside the loop body which had captured the external resource.

    Args:
      tensor: the external resource Tensor to be captured.

    Returns:
      Tensor in this graph.
    """
    assert tensor.dtype == dtypes.resource
    forward_graph_input_names = [t.name for t in self._forward_graph.inputs]
    forward_graph_name_to_opdef = {op.name: op.node_def for op in self._forward_graph.get_operations()}
    index = util.resource_input_index(tensor.name, forward_graph_input_names, forward_graph_name_to_opdef, self._forward_graph._functions)
    input_placeholder = self._forward_graph.inputs[index]
    tensor_in_outer_graph = self._forward_graph._while.inputs[index]
    assert input_placeholder.dtype == dtypes.resource
    assert tensor_in_outer_graph.dtype == dtypes.resource
    if index != util.resource_input_index(self._forward_graph.outputs[index].name, forward_graph_input_names, forward_graph_name_to_opdef, self._forward_graph._functions):
        raise AssertionError(f'Resource tensors must be loop invariants {tensor_in_outer_graph}')
    self._indirect_captures[ops.tensor_id(tensor)] = self.capture(tensor_in_outer_graph)
    return self._indirect_captures[ops.tensor_id(tensor)]