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
def _select_forward_and_backward_functions(self, args, possible_gradient_type, executing_eagerly):
    """Selects forward and backward functions based on the calling context.

    The forward function computes the "real" function outputs, `self._outputs`,
    and any extra values needed by the corresponding backward function.

    Args:
      args: A flat list of Tensors with all of the inputs to the forward
        function (including user-specified and captured inputs).
      possible_gradient_type: One of gradients_util.POSSIBLE_GRADIENT_TYPES_*.
      executing_eagerly: Boolean, the value of context.executing_eagerly().

    Returns:
      An object with a `forward` method returning a tuple of (forward_function :
      AtomicFunction, augmented_arguments : List), and a corresponding
      `record` method which takes outputs from the forward function and records
      the operation. forward_function should be called with augmented_arguments.
    """
    if executing_eagerly:
        input_tangents = forwardprop_util.pack_tangents(args)
    else:
        input_tangents = forwardprop_util.TangentInfo()
    need_gradients_for_jvps = record.should_record_backprop(input_tangents.tangents)
    cache_key = (need_gradients_for_jvps, input_tangents.indices)
    if possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER:
        if input_tangents.indices or executing_eagerly:
            functions = self._first_order_tape_functions.get(cache_key, None)
            if functions is None:
                functions = _FirstOrderTapeGradientFunctions(self._func_graph, self._attrs, self._garbage_collector, forwardprop_input_indices=input_tangents.indices, delayed_rewrite_functions=self._delayed_rewrite_functions, need_gradients_for_jvps=need_gradients_for_jvps)
                self._first_order_tape_functions[cache_key] = functions
            return _ForwardBackwardCall(functions, args, input_tangents.tangents, tape_watching=True)
        else:
            return _ForwardBackwardCall(self._delayed_rewrite_functions, args, input_tangents.tangents, tape_watching=True)
    elif possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER:
        functions = self._higher_order_tape_functions.get(cache_key, None)
        if functions is None:
            functions = _HigherOrderTapeGradientFunctions(self._func_graph, self._attrs, self._garbage_collector, forwardprop_input_indices=input_tangents.indices, delayed_rewrite_functions=self._delayed_rewrite_functions, need_gradients_for_jvps=need_gradients_for_jvps)
            self._higher_order_tape_functions[cache_key] = functions
        return _ForwardBackwardCall(functions, args, input_tangents.tangents, tape_watching=True)
    return _ForwardBackwardCall(self._delayed_rewrite_functions, args, input_tangents.tangents, tape_watching=False)