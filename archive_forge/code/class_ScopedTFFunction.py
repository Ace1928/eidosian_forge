import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class ScopedTFFunction(UniquePtr):
    """Wrapper around TF_Function that handles deletion."""

    def __init__(self, func, name):
        super(ScopedTFFunction, self).__init__(name=name, obj=func, deleter=c_api.TF_DeleteFunction)