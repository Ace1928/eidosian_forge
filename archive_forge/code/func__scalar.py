import numbers
import sys
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.util import tf_export
def _scalar(tf_fn, x, promote_to_float=False):
    """Computes the tf_fn(x) for each element in `x`.

  Args:
    tf_fn: function that takes a single Tensor argument.
    x: array_like. Could be an ndarray, a Tensor or any object that can be
      converted to a Tensor using `ops.convert_to_tensor`.
    promote_to_float: whether to cast the argument to a float dtype if it is not
      already.

  Returns:
    An ndarray with the same shape as `x`. The default output dtype is
    determined by `np_utils.result_type(float)`, unless x is an ndarray with a
    floating point type, in which case the output type is same as x.dtype.
  """
    x = np_array_ops.asarray(x)
    if promote_to_float and (not np.issubdtype(x.dtype.as_numpy_dtype, np.inexact)):
        x = x.astype(np_utils.result_type(float))
    return tf_fn(x)