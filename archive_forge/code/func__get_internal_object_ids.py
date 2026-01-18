import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
@trace.trace_wrapper
def _get_internal_object_ids(self):
    ids = set()
    for snapshot in self._snapshots:
        ids.add(id(snapshot))
        for v in snapshot.values():
            ids.add(id(v))
    return ids