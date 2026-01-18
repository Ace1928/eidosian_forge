import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def GetNumpyAppendFn(dtype):
    if dtype.type == np.bytes_ or dtype.type == np.str_:
        if _FAST_TENSOR_UTIL_AVAILABLE:
            return fast_tensor_util.AppendObjectArrayToTensorProto
        else:
            return SlowAppendObjectArrayToTensorProto
    return GetFromNumpyDTypeDict(_NP_TO_APPEND_FN, dtype)