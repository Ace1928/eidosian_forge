import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def _alloc_pid(self):
    """Allocate a process Id."""
    pid = self._next_pid
    self._next_pid += 1
    return pid