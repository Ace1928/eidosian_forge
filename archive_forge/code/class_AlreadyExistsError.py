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
@tf_export('errors.AlreadyExistsError')
class AlreadyExistsError(OpError):
    """Raised when an entity that we attempted to create already exists.

  An API raises this this error to avoid overwriting an existing resource,
  value, etc. Calling a creation API multiple times with the same arguments
  could raise this error if the creation API is not idempotent.

  For example, running an operation that saves a file
  (e.g. `tf.saved_model.save`)
  could potentially raise this exception if an explicit filename for an
  existing file was passed.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates an `AlreadyExistsError`."""
        super(AlreadyExistsError, self).__init__(node_def, op, message, ALREADY_EXISTS, *args)