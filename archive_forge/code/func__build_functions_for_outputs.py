import collections
import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _build_functions_for_outputs(self, outputs, inference_args, input_tangents):
    """Forward+backward functions where the backward function sees `outputs`."""
    trainable_outputs = []
    trainable_indices = []
    for index, output in enumerate(outputs):
        if backprop_util.IsTrainable(output):
            trainable_outputs.append(output)
            trainable_indices.append(index)
    backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
    with backwards_graph.as_default():
        gradients_wrt_outputs = []
        for output in trainable_outputs:
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(output)
            gradient_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            handle_data_util.copy_handle_data(output, gradient_placeholder)
            gradients_wrt_outputs.append(gradient_placeholder)
        with ops.device(None):
            gradients_wrt_inputs = gradients_util._GradientsHelper(trainable_outputs, self._func_graph.inputs, grad_ys=gradients_wrt_outputs, src_graph=self._func_graph)
        if input_tangents:
            gradients_wrt_inputs = nest.map_structure(lambda x: ops.convert_to_tensor(x) if x is not None else None, gradients_wrt_inputs)
        captures_from_forward = [c for c in backwards_graph.external_captures if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]
        existing_outputs = object_identity.ObjectIdentitySet(self._func_graph.outputs)
        for capture in captures_from_forward:
            if capture not in existing_outputs:
                existing_outputs.add(capture)
                self._func_graph.outputs.append(capture)
    backwards_graph.inputs = gradients_wrt_outputs + backwards_graph.internal_captures
    backwards_graph.outputs.extend((grad for grad in nest.flatten(gradients_wrt_inputs, expand_composites=True) if grad is not None))
    backwards_graph.structured_outputs = gradients_wrt_inputs
    forward_function, backward_function = _create_forward_backward_with_graph(self._attrs, self._func_graph, backwards_graph)
    if not input_tangents:
        return (forward_function, self._func_graph, backward_function, None, 0)
    forward_wrapper = self._wrap_forward_function_with_jvps(forward_function, backward_function, inference_args, input_tangents)
    wrapped_backwards_graph, forward_wrapper = self._wrap_backward_function_with_jvp_backprop(backward_function, gradients_wrt_outputs, forward_wrapper)
    forward_wrapper = self._shuffle_forward_outputs(forward_wrapper)
    wrapped_forward_function, wrapped_backward_function = _create_forward_backward_with_graph(self._attrs, forward_wrapper.graph, wrapped_backwards_graph)
    if len(inference_args) + len(input_tangents) != len(forward_wrapper.graph.inputs):
        raise errors.InternalError(f'The forward graph had {len(forward_wrapper.graph.inputs)} inputs, but we expected {len(inference_args) + len(input_tangents)} ({len(inference_args)} inference inputs and {len(input_tangents)} input tangents).')
    return (wrapped_forward_function, forward_wrapper.graph, wrapped_backward_function, forward_wrapper.output_indices, len(forward_wrapper.output_tangents))