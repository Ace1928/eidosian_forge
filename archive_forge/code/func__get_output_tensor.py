import collections
import threading
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
def _get_output_tensor(self, op_type, tensor, checked_tensor, is_v1_graph_mode):
    """Determine what tensor to output from callback.

    Args:
      op_type: Type of the op that outputs the original symbolic tensor, as
        `bytes`.
      tensor: The original output symbolic tensor.
      checked_tensor: The debugger-instrumented, numerics-checking tensor.
      is_v1_graph_mode: Whether the debugged proggram is running under V1 graph
        mode.

    Returns:
      A symbolic tensor to be returned by the dumping op_callback.
    """
    if is_v1_graph_mode:
        if op_type == b'Placeholder':
            self._placeholder_to_debug_tensor[tensor] = checked_tensor
            return tensor
        else:
            return checked_tensor
    else:
        return tensor