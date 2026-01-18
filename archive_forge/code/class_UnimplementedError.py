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
@tf_export('errors.UnimplementedError')
class UnimplementedError(OpError):
    """Raised when an operation has not been implemented.

  Some operations may raise this error when passed otherwise-valid
  arguments that it does not currently support. For example, running
  the `tf.nn.max_pool2d` operation
  would raise this error if pooling was requested on the batch dimension,
  because this is not yet supported.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates an `UnimplementedError`."""
        super(UnimplementedError, self).__init__(node_def, op, message, UNIMPLEMENTED, *args)