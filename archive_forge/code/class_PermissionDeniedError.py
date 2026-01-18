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
@tf_export('errors.PermissionDeniedError')
class PermissionDeniedError(OpError):
    """Raised when the caller does not have permission to run an operation.

  For example, running the
  `tf.WholeFileReader.read`
  operation could raise `PermissionDeniedError` if it receives the name of a
  file for which the user does not have the read file permission.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates a `PermissionDeniedError`."""
        super(PermissionDeniedError, self).__init__(node_def, op, message, PERMISSION_DENIED, *args)