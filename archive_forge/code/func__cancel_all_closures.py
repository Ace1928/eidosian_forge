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
def _cancel_all_closures(self):
    """Clears the queue and sets remaining closures cancelled error.

    This method expects self._queue_lock to be held prior to entry.
    """
    self._cancellation_mgr.start_cancel()
    logging.info('Canceling all closures: waiting for inflight closures to finish')
    while self._inflight_closure_count > 0:
        self._no_inflight_closure_condition.wait()
    logging.info('Canceling all closures: canceling remaining closures on the queue')
    while True:
        try:
            closure = self._queue.get(block=False)
            metric_utils.monitor_int('queued_closures', self._queue.qsize())
            self._queue_free_slot_condition.notify()
            closure.mark_cancelled()
        except queue.Empty:
            break
    self._cancellation_mgr = cancellation.CancellationManager()