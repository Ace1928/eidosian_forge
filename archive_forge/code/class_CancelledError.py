import traceback
import warnings
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python import _pywrap_py_exception_registry
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('errors.CancelledError')
class CancelledError(OpError):
    """Raised when an operation is cancelled.

  For example, a long-running operation e.g.`tf.queue.QueueBase.enqueue`, or a
  `tf.function` call may be cancelled by either running another operation e.g.
  `tf.queue.QueueBase.close` or a remote worker failure.

  This long-running operation will fail by raising `CancelledError`.

  Example:
  >>> q = tf.queue.FIFOQueue(10, tf.float32, ((),))
  >>> q.enqueue((10.0,))
  >>> q.close()
  >>> q.enqueue((10.0,))
  Traceback (most recent call last):
    ...
  CancelledError: ...

  """

    def __init__(self, node_def, op, message, *args):
        """Creates a `CancelledError`."""
        super(CancelledError, self).__init__(node_def, op, message, CANCELLED, *args)