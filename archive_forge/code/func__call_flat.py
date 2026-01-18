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
def _call_flat(self, tensor_inputs, captured_inputs):
    """Executes the wrapped function.

    Args:
      tensor_inputs: a list of only Tensors generated from args, kwargs.
      captured_inputs: the captured inputs that are also part of the input args
        to the actual execution. By default, it should be self._captured_inputs.
    Returns:
      The result of applying the TF function to `args`.

    Raises:
      ValueError: If `args` contains anything other than Tensors or Variables.
    """
    ctx = context.context()
    executing_eagerly = ctx.executing_eagerly()
    default_graph = ops.get_default_graph()
    if default_graph.building_function and (not self._func_graph.saveable):
        default_graph.mark_as_unsaveable(self._func_graph.saving_errors)
    if record.could_possibly_record() or hasattr(default_graph, 'watch_variable'):
        for v in self._func_graph.variables:
            resource_variable_ops.variable_accessed(v)
    if not executing_eagerly:
        for i, tensor_input in enumerate(tensor_inputs):
            if tensor_input.dtype == dtypes.resource or tensor_input.dtype == dtypes.variant:
                continue
            graph_input_shape = tensor_shape.TensorShape(self._func_graph.inputs[i].shape)
            if not graph_input_shape.is_compatible_with(tensor_input.shape):
                raise ValueError(f'Tensor {tensor_input} is not compatible with the shape this function was traced with. Expected shape {self._func_graph.inputs[i].shape}, but got shape {tensor_input.shape}.\n\nIf you called get_concrete_function, you may need to pass a tf.TensorSpec(..., shape=...) with a less specific shape, having None on axes which can vary.')
    args = tensor_inputs + captured_inputs
    possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
    if possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE and executing_eagerly:
        return self._inference_function.flat_call(args)
    forward_backward = self._select_forward_and_backward_functions(args, possible_gradient_type, executing_eagerly)
    forward_function, args_with_tangents = forward_backward.forward()
    if executing_eagerly:
        flat_outputs = forward_function(*args_with_tangents)
    else:
        with default_graph._override_gradient_function({'PartitionedCall': self._get_gradient_function(), 'StatefulPartitionedCall': self._get_gradient_function()}):
            flat_outputs = forward_function(*args_with_tangents)
    forward_backward.record(flat_outputs)
    return self.function_type.pack_output(flat_outputs)