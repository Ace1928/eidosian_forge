import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class AlreadyGarbageCollectedError(Exception):

    def __init__(self, name, obj_type):
        super(AlreadyGarbageCollectedError, self).__init__(f'{name} of type {obj_type} has already been garbage collected and cannot be called.')