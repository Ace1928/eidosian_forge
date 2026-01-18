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
@tf_export(v1=['errors.raise_exception_on_not_ok_status'])
class raise_exception_on_not_ok_status(object):
    """Context manager to check for C API status."""

    def __enter__(self):
        self.status = c_api_util.ScopedTFStatus()
        return self.status.status

    def __exit__(self, type_arg, value_arg, traceback_arg):
        try:
            if c_api.TF_GetCode(self.status.status) != 0:
                raise _make_specific_exception(None, None, compat.as_text(c_api.TF_Message(self.status.status)), c_api.TF_GetCode(self.status.status))
        finally:
            del self.status
        return False