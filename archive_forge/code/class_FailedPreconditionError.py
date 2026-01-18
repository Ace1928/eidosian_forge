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
@tf_export('errors.FailedPreconditionError')
class FailedPreconditionError(OpError):
    """Raised when some prerequisites are not met when running an operation.

  This typically indicates that system is not in state to execute the operation
  and requires preconditions to be met before successfully executing current
  operation.

  For example, this exception is commonly raised when running an operation
  that reads a `tf.Variable` before it has been initialized.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates a `FailedPreconditionError`."""
        super(FailedPreconditionError, self).__init__(node_def, op, message, FAILED_PRECONDITION, *args)