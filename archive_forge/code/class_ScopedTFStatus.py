import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class ScopedTFStatus(object):
    """Wrapper around TF_Status that handles deletion."""
    __slots__ = ['status']

    def __init__(self):
        self.status = c_api.TF_NewStatus()

    def __del__(self):
        if c_api is not None and c_api.TF_DeleteStatus is not None:
            c_api.TF_DeleteStatus(self.status)