import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class ScopedTFBuffer(object):
    """An internal class to help manage the TF_Buffer lifetime."""
    __slots__ = ['buffer']

    def __init__(self, buf_string):
        self.buffer = c_api.TF_NewBufferFromString(compat.as_bytes(buf_string))

    def __del__(self):
        c_api.TF_DeleteBuffer(self.buffer)