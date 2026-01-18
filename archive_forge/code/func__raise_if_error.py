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
def _raise_if_error(self):
    """Raises the error if one exists.

    If an error exists, cancel the closures in queue, raises it, and clear
    the error.

    This method expects self._queue_lock to be held prior to entry.
    """
    if self._error:
        logging.error('Start cancelling closures due to error %r: %s', self._error, self._error)
        self._cancel_all_closures()
        try:
            raise self._error
        finally:
            self._error = None