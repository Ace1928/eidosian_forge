import collections
import copy
import gc
from tensorflow.python.framework import _python_memory_checker_helper
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
def _record_snapshot():
    self._snapshots.append(_create_python_object_snapshot())