import numpy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base as trackable
def _serialize_to_proto(self, object_proto=None, **kwargs):
    object_proto.constant.operation = self._exported_tensor.op.name