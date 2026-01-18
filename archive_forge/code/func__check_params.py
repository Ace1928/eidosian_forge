import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _check_params(window_length, dtype):
    """Check window_length and dtype params.

  Args:
    window_length: A scalar value or `Tensor`.
    dtype: The data type to produce. Must be a floating point type.

  Returns:
    window_length converted to a tensor of type int32.

  Raises:
    ValueError: If `dtype` is not a floating point type or window_length is not
      a scalar.
  """
    if not dtype.is_floating:
        raise ValueError('dtype must be a floating point type. Found %s' % dtype)
    window_length = ops.convert_to_tensor(window_length, dtype=dtypes.int32)
    window_length.shape.assert_has_rank(0)
    return window_length