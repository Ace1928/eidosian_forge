import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_trace_mode(self):
    """Checks if the given trace mode is valid."""
    found, trace_mode = self.get_flag_value(FLAG_NAME_TRACE_MODE)
    if not found or not trace_mode:
        trace_mode = TRACE_MODE_NORM
    valid_trace_modes = [TRACE_MODE_NAN_INF, TRACE_MODE_PART_TENSOR, TRACE_MODE_FULL_TENSOR, TRACE_MODE_NORM, TRACE_MODE_MAX_ABS, TRACE_MODE_SUMMARY, TRACE_MODE_FULL_TENSOR_SUMMARY, TRACE_MODE_HISTORY]
    if trace_mode not in valid_trace_modes:
        raise ValueError('Invalid trace mode "%s" given to the Tensor_Tracer.Valid trace modes are: %s' % (trace_mode, valid_trace_modes))
    return trace_mode