import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def _asset_path_from_tensor(path_tensor):
    """Returns the filepath value stored in constant `path_tensor`.

  Args:
    path_tensor: Tensor of a file-path.

  Returns:
    The string value i.e. path of the tensor, if valid.

  Raises:
    TypeError if tensor does not match expected op type, dtype or value.
  """
    if not isinstance(path_tensor, tensor.Tensor):
        raise TypeError(f'Asset path tensor {path_tensor} must be a Tensor.')
    if path_tensor.op.type != 'Const':
        raise TypeError(f'Asset path tensor {path_tensor} must be of type constant.Has type {path_tensor.op.type} instead.')
    if path_tensor.dtype != dtypes.string:
        raise TypeError(f'Asset path tensor {path_tensor}` must be of dtype string.Has type {path_tensor.dtype} instead.')
    str_values = path_tensor.op.get_attr('value').string_val
    if len(str_values) != 1:
        raise TypeError(f'Asset path tensor {path_tensor} must be a scalar.')
    return str_values[0]