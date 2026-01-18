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
def _jvp_dispatch(op_name, attr_tuple, inputs, outputs, tangents, use_batch=False):
    """Determine which forwardprop function to call."""
    if _TRACE_COUNT.get(op_name, 0) < _TRACE_COUNT_LIMIT:
        config = _jvp_exact_config
    else:
        config = _jvp_relaxed_config
    return tracing_compilation.call_function((op_name, attr_tuple, inputs, outputs, tangents, use_batch), tracing_options=config)