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
class _ForwardBackwardCall(object):
    """Holds the state of a function call between execution and recording."""
    __slots__ = ['_functions', '_inference_args', '_input_tangents', '_tape_watching']

    def __init__(self, functions, inference_args, input_tangents, tape_watching):
        """Collects information about the function call.

    Args:
      functions: An object which produces forward and backward functions, either
        a _DelayedRewriteGradientFunctions or a _TapeGradientFunctions object.
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.
      tape_watching: Boolean, with True indicating that recording is necessary.
    """
        self._functions = functions
        self._inference_args = inference_args
        self._input_tangents = input_tangents
        self._tape_watching = tape_watching

    def forward(self):
        """Builds or retrieves a forward function for this call."""
        forward_function = self._functions.forward(self._inference_args, self._input_tangents)
        return (forward_function, self._inference_args + self._input_tangents)

    def record(self, flat_outputs):
        """Given outputs from the execution of `forward`, records the operation."""
        if self._tape_watching and (not isinstance(flat_outputs, ops.Operation)) and (flat_outputs is not None):
            self._functions.record(flat_outputs, self._inference_args, self._input_tangents)