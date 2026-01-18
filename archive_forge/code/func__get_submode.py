import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def _get_submode(self):
    """Checks if the given submode is valid."""
    found, submode = self.get_flag_value(FLAG_NAME_SUBMODE)
    if not found or not submode:
        submode = _SUBMODE_DETAILED
    if not submode:
        return
    valid_submodes = [_SUBMODE_DETAILED, _SUBMODE_BRIEF]
    if submode not in valid_submodes:
        raise ValueError('Invalid submode "%s" given to the Tensor_Tracer.Valid submodes are: %s' % (submode, valid_submodes))
    return submode