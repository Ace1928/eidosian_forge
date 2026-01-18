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
@tf_export('errors.AbortedError')
class AbortedError(OpError):
    """Raised when an operation was aborted, typically due to a concurrent action.

  For example, running a
  `tf.queue.QueueBase.enqueue`
  operation may raise `AbortedError` if a
  `tf.queue.QueueBase.close` operation
  previously ran.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates an `AbortedError`."""
        super(AbortedError, self).__init__(node_def, op, message, ABORTED, *args)