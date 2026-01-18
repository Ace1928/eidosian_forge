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
@tf_export('errors.DataLossError')
class DataLossError(OpError):
    """Raised when unrecoverable data loss or corruption is encountered.

  This could be due to:
  * A truncated file.
  * A corrupted file.
  * Specifying the wrong data format.

  For example, this may be raised by running a
  `tf.WholeFileReader.read`
  operation, if the file is truncated while it is being read.
  """

    def __init__(self, node_def, op, message, *args):
        """Creates a `DataLossError`."""
        super(DataLossError, self).__init__(node_def, op, message, DATA_LOSS, *args)