import atexit
import collections
import copy
import queue
import threading
import time
import weakref
from absl import logging
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import ops
from tensorflow.python.ops.resource_variable_ops import UninitializedVariable
from tensorflow.python.ops.variables import Variable
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.util import object_identity
def _join_async_save_thread(self):
    """Join the async save thread.

    The steps for terminating the async save thread:
    1). Put will succeed when the last async save event is done. Putting a false
        triggers the async save thread's while loop to end. We use put instead
        of sync because sync does not have a timeout argument.
    2). Join the async save thread. (The thread may finish before joining.)
    """
    try:
        self._queue.put(False, timeout=300)
        logging.info('Joining the async save thread.')
        if self._async_save_thread is not None:
            self._async_save_thread.join()
    except queue.Full:
        logging.error('Timeout waiting for the async save thread; terminating the thread instead. The last checkpoint may be incomeplete.')
    finally:
        self._check_async_thread_error()