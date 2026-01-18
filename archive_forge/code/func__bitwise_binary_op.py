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
def _bitwise_binary_op(tf_fn, x1, x2):

    def f(x1, x2):
        is_bool = x1.dtype == dtypes.bool
        if is_bool:
            assert x2.dtype == dtypes.bool
            x1 = math_ops.cast(x1, dtypes.int8)
            x2 = math_ops.cast(x2, dtypes.int8)
        r = tf_fn(x1, x2)
        if is_bool:
            r = math_ops.cast(r, dtypes.bool)
        return r
    return _bin_op(f, x1, x2)