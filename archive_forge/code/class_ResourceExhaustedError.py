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
@tf_export('errors.ResourceExhaustedError')
class ResourceExhaustedError(OpError):
    """Raised when some resource has been exhausted while running operation.

  For example, this error might be raised if a per-user quota is
  exhausted, or perhaps the entire file system is out of space. If running into
  `ResourceExhaustedError` due to out of memory (OOM), try to use smaller batch
  size or reduce dimension size of model weights.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates a `ResourceExhaustedError`."""
        super(ResourceExhaustedError, self).__init__(node_def, op, message, RESOURCE_EXHAUSTED, *args)