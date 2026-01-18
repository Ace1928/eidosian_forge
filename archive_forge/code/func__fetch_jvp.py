import functools
import threading
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import execute
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _fetch_jvp(tensor):
    if hasattr(tensor, 'handle'):
        unwrapped_tensor = ops.convert_to_tensor(tensor.handle)
    else:
        unwrapped_tensor = tensor
    result = pywrap_tfe.TFE_Py_ForwardAccumulatorJVP(self._accumulator, unwrapped_tensor)
    if result is None and unconnected_gradients == UnconnectedGradients.ZERO:
        result = array_ops.zeros_like(tensor)
    return result