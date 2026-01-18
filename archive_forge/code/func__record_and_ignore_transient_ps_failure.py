import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _record_and_ignore_transient_ps_failure(self, e):
    """Records potential PS failures and return if failure should be ignored."""
    if self._transient_ps_failures_threshold <= 0 or not _is_ps_failure(e):
        return False
    ps_tasks = _extract_failed_ps_instances(str(e))
    with self._potential_ps_failures_lock:
        for t in ps_tasks:
            self._potential_ps_failures_count[t] += 1
            if self._potential_ps_failures_count[t] >= self._transient_ps_failures_threshold:
                return False
    return True