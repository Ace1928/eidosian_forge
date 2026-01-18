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
def _capture_helper(self, tensor, name):
    """Implements the capturing described in the class docstring."""
    captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
    if captured_tensor is not None:
        return captured_tensor
    if tensor.graph is not self._forward_graph:
        already_captured = id(tensor) in self.function_captures.by_val_internal
        captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)
        if not already_captured:
            self.internal_capture_to_output[ops.tensor_id(captured_tensor)] = captured_tensor
            self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
        return captured_tensor
    while tensor.op.type == 'Identity':
        tensor = tensor.op.inputs[0]
    captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
    if captured_tensor is not None:
        return captured_tensor
    if _is_loop_invariant(tensor, self._forward_graph.inputs, self._forward_graph.outputs):
        captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)
        self.internal_capture_to_output[ops.tensor_id(captured_tensor)] = captured_tensor
        self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
        return captured_tensor
    if constant_op.is_constant(tensor):
        real_value = constant_op.constant(tensor_util.constant_value(tensor), dtype=tensor.dtype)
        self._indirect_captures[ops.tensor_id(tensor)] = real_value
        return real_value
    if tensor.dtype == dtypes.resource:
        return self._resource_capture_helper(tensor)
    accumulator = _get_accumulator(tensor)
    if accumulator is None:
        with self._forward_graph.outer_graph.as_default():
            with util.clear_control_inputs():
                tensor_list = list_ops.empty_tensor_list(element_dtype=tensor.dtype, element_shape=tensor.shape, max_num_elements=self._maximum_iterations, name=_build_accumulator_name(tensor))
        self.extra_inputs.append(tensor_list)
        with self._forward_graph.as_default():
            accumulator = list_ops.tensor_list_push_back(tensor_list, tensor)
        self._forward_graph.outputs.append(accumulator)
        with self._forward_cond_graph.as_default():
            self._forward_cond_graph.capture(tensor_list)
    captured_accumulator = super(_WhileBodyGradFuncGraph, self)._capture_helper(accumulator, name)
    new_tensor_list, captured_tensor = list_ops.tensor_list_pop_back(captured_accumulator, element_dtype=tensor.dtype)
    self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
    self.internal_capture_to_output[ops.tensor_id(captured_accumulator)] = new_tensor_list
    return captured_tensor