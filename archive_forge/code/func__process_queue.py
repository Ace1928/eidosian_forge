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
def _process_queue(self):
    """Function running in a worker thread to process closure queues."""
    self._maybe_delay()
    while self._should_worker_thread_run:
        closure = self._cluster.closure_queue.get(tag=self.worker_index)
        if not self._should_worker_thread_run or closure is None:
            if closure is not None:
                closure.mark_cancelled()
            return
        if isinstance(closure, ResourceClosure):
            self._process_resource_closure(closure)
        else:
            self._process_closure(closure)
        del closure