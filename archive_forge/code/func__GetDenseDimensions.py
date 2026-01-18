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
def _GetDenseDimensions(list_of_lists):
    """Returns the inferred dense dimensions of a list of lists."""
    if not isinstance(list_of_lists, (list, tuple)):
        return []
    elif not list_of_lists:
        return [0]
    else:
        return [len(list_of_lists)] + _GetDenseDimensions(list_of_lists[0])