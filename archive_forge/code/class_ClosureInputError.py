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
class ClosureInputError(Exception):
    """Wrapper for errors from resource building.

  When a closure starts, it first checks for errors in any of its inputs, which
  are RemoteValues from resource closures. If there were any errors, it wraps
  the exception in this class and raises so it can be handled by the worker
  failure handler.

  Attributes:
    original_exception:
  """

    def __init__(self, original_exception):
        if isinstance(original_exception, (ClosureInputError, ClosureAbortedError)):
            self.original_exception = original_exception.original_exception
        else:
            self.original_exception = original_exception
        message = 'Input has an error, the original exception is %r, error message is %s.' % (self.original_exception, str(self.original_exception))
        super().__init__(message)
        self.with_traceback(original_exception.__traceback__)