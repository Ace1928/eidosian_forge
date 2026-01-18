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
def _on_resource_closure_failure(self, e):
    """Clear tagged queue to ensure resource closures are rebuilt.

    Args:
      e: The exception arisen from the resource closure.
    """
    logging.info('[Worker %d] Clearing tagged queue after resource closure failure.', self.worker_index)
    with self._resource_tracking_lock:
        self._is_dead_with_error = e
        self._cluster.closure_queue.clear_tag_unlocked(self.worker_index)
        self._set_resources_aborted(e)