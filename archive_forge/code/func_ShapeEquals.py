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
def ShapeEquals(tensor_proto, shape):
    """Returns True if "tensor_proto" has the given "shape".

  Args:
    tensor_proto: A TensorProto.
    shape: A tensor shape, expressed as a TensorShape, list, or tuple.

  Returns:
    True if "tensor_proto" has the given "shape", otherwise False.

  Raises:
    TypeError: If "tensor_proto" is not a TensorProto, or shape is not a
      TensorShape, list, or tuple.
  """
    if not isinstance(tensor_proto, tensor_pb2.TensorProto):
        raise TypeError(f'`tensor_proto` must be a tensor_pb2.TensorProto object, but got type {type(tensor_proto)}.')
    if isinstance(shape, tensor_shape_pb2.TensorShapeProto):
        shape = [d.size for d in shape.dim]
    elif not isinstance(shape, (list, tuple)):
        raise TypeError(f'`shape` must be a list or tuple, but got type {type(shape)}.')
    tensor_shape_list = [d.size for d in tensor_proto.tensor_shape.dim]
    return all((x == y for x, y in zip(tensor_shape_list, shape)))