import numpy as np
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops.gen_list_ops import *
def _build_element_shape(shape):
    """Converts shape to a format understood by list_ops for element_shape.

  If `shape` is already a `Tensor` it is returned as-is. We do not perform a
  type check here.

  If shape is None or a TensorShape with unknown rank, -1 is returned.

  If shape is a scalar, an int32 tensor with empty list is returned. Note we
  do directly return an empty list since ops.convert_to_tensor would conver it
  to a float32 which is not a valid type for element_shape.

  If shape is a sequence of dims, None's in the list are replaced with -1. We
  do not check the dtype of the other dims.

  Args:
    shape: Could be None, Tensor, TensorShape or a list of dims (each dim could
      be a None, scalar or Tensor).

  Returns:
    A None-free shape that can be converted to a tensor.
  """
    if isinstance(shape, tensor_lib.Tensor):
        return shape
    if isinstance(shape, tensor_shape.TensorShape):
        shape = shape.as_list() if shape else None
    if shape is None:
        return -1
    if isinstance(shape, (np.ndarray, np.generic)) or not shape:
        return ops.convert_to_tensor(shape, dtype=dtypes.int32)

    def convert(val):
        if val is None:
            return -1
        if isinstance(val, tensor_lib.Tensor):
            return val
        if isinstance(val, tensor_shape.Dimension):
            return val.value if val.value is not None else -1
        return val
    return [convert(d) for d in shape]