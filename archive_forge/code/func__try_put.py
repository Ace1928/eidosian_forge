import collections
import os.path
import sys
import threading
import time
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _try_put(self, item):
    """Attempts to enqueue an item to the event queue.

    If the queue is closed, this will close the EventFileWriter and reraise the
    exception that caused the queue closure, if one exists.

    Args:
      item: the item to enqueue
    """
    try:
        self._event_queue.put(item)
    except QueueClosedError:
        self._internal_close()
        if self._worker.failure_exc_info:
            _, exception, _ = self._worker.failure_exc_info
            raise exception from None