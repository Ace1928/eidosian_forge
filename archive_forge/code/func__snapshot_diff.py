import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
@trace.trace_wrapper
def _snapshot_diff(self, old_index, new_index):
    return _snapshot_diff(self._snapshots[old_index], self._snapshots[new_index], self._get_internal_object_ids())