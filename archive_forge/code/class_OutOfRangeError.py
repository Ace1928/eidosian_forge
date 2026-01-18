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
@tf_export('errors.OutOfRangeError')
class OutOfRangeError(OpError):
    """Raised when an operation iterates past the valid range.

  Unlike `InvalidArgumentError`, this error indicates a problem may be fixed if
  the system state changes. For example, if a list grows and the operation is
  now within the valid range. `OutOfRangeError` overlaps with
  `FailedPreconditionError` and should be preferred as the more specific error
  when iterating or accessing a range.

  For example, iterating a TF dataset past the last item in the dataset will
  raise this error.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates an `OutOfRangeError`."""
        super(OutOfRangeError, self).__init__(node_def, op, message, OUT_OF_RANGE, *args)