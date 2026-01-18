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
def _tf_gcd(x1, x2):

    def _gcd_cond_fn(_, x2):
        return math_ops.reduce_any(x2 != 0)

    def _gcd_body_fn(x1, x2):
        x2_safe = array_ops.where_v2(x2 != 0, x2, constant_op.constant(1, x2.dtype))
        x1, x2 = (array_ops.where_v2(x2 != 0, x2, x1), array_ops.where_v2(x2 != 0, math_ops.mod(x1, x2_safe), constant_op.constant(0, x2.dtype)))
        return (array_ops.where_v2(x1 < x2, x2, x1), array_ops.where_v2(x1 < x2, x1, x2))
    if not np.issubdtype(x1.dtype.as_numpy_dtype, np.integer) or not np.issubdtype(x2.dtype.as_numpy_dtype, np.integer):
        raise ValueError('Arguments to gcd must be integers.')
    shape = array_ops.broadcast_dynamic_shape(array_ops.shape(x1), array_ops.shape(x2))
    x1 = array_ops.broadcast_to(x1, shape)
    x2 = array_ops.broadcast_to(x2, shape)
    value, _ = while_loop.while_loop(_gcd_cond_fn, _gcd_body_fn, (math_ops.abs(x1), math_ops.abs(x2)))
    return value