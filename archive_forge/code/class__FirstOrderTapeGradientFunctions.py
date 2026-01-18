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
class _FirstOrderTapeGradientFunctions(_TapeGradientFunctions):
    """Caches tape-friendly functions for first-order gradients."""

    def __init__(self, func_graph, attrs, func_graph_deleter, forwardprop_input_indices, delayed_rewrite_functions, need_gradients_for_jvps):
        super().__init__(func_graph, attrs, func_graph_deleter, forwardprop_input_indices, delayed_rewrite_functions, need_gradients_for_jvps)
        self._func_graph_deleter = func_graph_deleter
        self._forwardprop_input_indices = forwardprop_input_indices

    def _forward_and_backward_functions(self, inference_args, input_tangents):
        """Shortcut for when only first-order gradients are required.

    The returned backward function does not accept gradients with respect to
    side output of forward_function. This is fine as long as the user can't
    possibly request second order tape gradients, as when they've used a single
    non-persistent GradientTape. Since we don't need the backward function to
    take gradients with respect to side outputs, we can skip some potentially
    slow graph building.

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
          gradients with respect to the "real" outputs of forward_function and
          returns gradients with respect to the inputs.
    """
        outputs = self._func_graph.outputs[:self._num_inference_outputs]
        return self._build_functions_for_outputs(outputs, inference_args, input_tangents)