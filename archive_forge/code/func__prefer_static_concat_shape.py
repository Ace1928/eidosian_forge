from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.util.tf_export import tf_export
def _prefer_static_concat_shape(first_shape, second_shape_int_list):
    """Concatenate a shape with a list of integers as statically as possible.

  Args:
    first_shape: `TensorShape` or `Tensor` instance. If a `TensorShape`,
      `first_shape.is_fully_defined()` must return `True`.
    second_shape_int_list: `list` of scalar integer `Tensor`s.

  Returns:
    `Tensor` representing concatenating `first_shape` and
      `second_shape_int_list` as statically as possible.
  """
    second_shape_int_list_static = [tensor_util.constant_value(s) for s in second_shape_int_list]
    if isinstance(first_shape, tensor_shape.TensorShape) and all((s is not None for s in second_shape_int_list_static)):
        return first_shape.concatenate(second_shape_int_list_static)
    return array_ops.concat([first_shape, second_shape_int_list], axis=0)