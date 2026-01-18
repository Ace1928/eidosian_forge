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
class _HigherOrderTapeGradientFunctions(_TapeGradientFunctions):
    """Caches tape-friendly functions for higher-order gradients."""

    def _forward_and_backward_functions(self, inference_args, input_tangents):
        """Forward and backward functions suitable for higher-order gradients.

    Unlike in `_FirstOrderTapeGradientFunctions`, the backward function built by
    this method accepts gradients for all of the outputs of the returned forward
    function, including side outputs.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to all of its outputs, real and side. Returns
          gradients with respect to the inputs.
    """
        outputs = []
        iteration_count = 0
        while len(outputs) < len(self._func_graph.outputs) and any((backprop_util.IsTrainable(output) for output in self._func_graph.outputs[len(outputs):])):
            iteration_count += 1
            if iteration_count >= 20 and iteration_count % 5 == 0:
                new_op_with_trainable_output = None
                num_new_trainable_outputs = 0
                for output in self._func_graph.outputs[len(outputs):]:
                    if backprop_util.IsTrainable(output):
                        num_new_trainable_outputs += 1
                        new_op_with_trainable_output = output.op
                logging.warning("Determining side outputs for the function '{}' is taking longer than expected ({} iterations, typically this converges in 5 or so). This could indicate that a gradient registration is adding new ops to the forward pass every time gradients are generated. {} new trainable output(s) were added this iteration, one from the following op:\n {}\nThis may indicate a TensorFlow bug, or an issue in a tf.custom_gradient.".format(self._func_graph.name, iteration_count, num_new_trainable_outputs, new_op_with_trainable_output))
            outputs = list(self._func_graph.outputs)
            self._build_functions_for_outputs(outputs, inference_args, input_tangents)
        forward_function, forward_graph, backward_function, output_indices, num_output_tangents = self._build_functions_for_outputs(outputs, inference_args, input_tangents)
        if len(self._func_graph.outputs) > len(outputs) and any((backprop_util.IsTrainable(output) for output in self._func_graph.outputs[len(outputs):])):
            raise errors.InternalError(f'Unexpectedly added new outputs to the forward function when building the backward function: {self._func_graph.outputs[len(outputs):]}.')
        return (forward_function, forward_graph, backward_function, output_indices, num_output_tangents)