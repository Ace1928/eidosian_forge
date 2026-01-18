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
class _CondGradFuncGraph(util.CondBranchFuncGraph):
    """FuncGraph for the gradient function of the branch of an If op.

  Handles wrapping and unwrapping intermediate values that are captured by the
  gradient computation in optionals.

  Attributes:
    op_needs_rewrite: True if any intermediates were captured, meaning the
      forward If op needs to be written to output the wrapped intermediates.
  """

    def __init__(self, name, forward_graph):
        super(_CondGradFuncGraph, self).__init__(name, collections=ops.get_default_graph()._collections)
        self.op_needs_rewrite = False
        self._forward_graph = forward_graph
        self._indirect_captures = {}
        self._wrapped_intermediates = collections.OrderedDict()
        self._xla_intermediates = []
        self._captured_constants = {}

    @property
    def wrapped_intermediates(self):
        """The optional-wrapped intermediates captured from the forward graph."""
        return list(self._wrapped_intermediates.values())

    @property
    def xla_intermediates(self):
        """Raw intermediates captured from the forward graph if XLA is enabled."""
        return self._xla_intermediates

    def _capture_helper(self, tensor, name):
        if tensor.graph is not self._forward_graph or any((tensor is t for t in self._forward_graph.inputs)) or any((tensor is t for t in self._forward_graph.outputs)):
            return super(_CondGradFuncGraph, self)._capture_helper(tensor, name)
        tensor_id = ops.tensor_id(tensor)
        if tensor_id in self._captured_constants:
            return self._captured_constants[tensor_id]
        elif constant_op.is_constant(tensor):
            self._captured_constants[tensor_id] = constant_op.constant(tensor_util.constant_value(tensor), dtype=tensor.dtype)
            return self._captured_constants[tensor_id]
        if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            if all((tensor is not capture for capture in self.external_captures)):
                self.xla_intermediates.append(tensor)
                self.op_needs_rewrite = True
            return super(_CondGradFuncGraph, self)._capture_helper(tensor, name)
        captured_tensor = self._indirect_captures.get(tensor_id)
        if captured_tensor is not None:
            return captured_tensor
        if tensor.dtype == dtypes.resource:
            index = util.resource_input_index(tensor.name, [t.name for t in self._forward_graph.inputs], {op.name: op.node_def for op in self._forward_graph.get_operations()}, self._forward_graph._functions)
            captured_tensor = super(_CondGradFuncGraph, self)._capture_helper(self._forward_graph.inputs[index], name)
        else:
            if tensor_id not in self._wrapped_intermediates:
                for consumer in tensor.consumers():
                    if consumer.type == 'OptionalFromValue' and any((consumer.outputs[0] is output for output in self._forward_graph.outputs)):
                        optional = consumer.outputs[0]
                        break
                else:
                    with self._forward_graph.as_default():
                        optional = gen_optional_ops.optional_from_value([tensor])
                    self.op_needs_rewrite = True
                self._wrapped_intermediates[tensor_id] = optional
            optional = self._wrapped_intermediates[tensor_id]
            captured_optional = super(_CondGradFuncGraph, self)._capture_helper(optional, name)
            captured_tensor = gen_optional_ops.optional_get_value(captured_optional, [tensor.dtype], [tensor.shape])[0]
        self._indirect_captures[tensor_id] = captured_tensor
        return captured_tensor