import collections
import os
import re
import site
import traceback
from tensorflow.python.util import tf_stack
def _compute_device_assignment_summary_from_op(op, prefix=''):
    return _compute_device_summary_from_list(op.name, op._device_assignments, prefix)