import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.signal import dct_ops
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.ops.signal import reconstruction_ops
from tensorflow.python.ops.signal import shape_ops
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _enclosing_power_of_two(value):
    """Return 2**N for integer N such that 2**N >= value."""
    value_static = tensor_util.constant_value(value)
    if value_static is not None:
        return constant_op.constant(int(2 ** np.ceil(np.log(value_static) / np.log(2.0))), value.dtype)
    return math_ops.cast(math_ops.pow(2.0, math_ops.ceil(math_ops.log(math_ops.cast(value, dtypes.float32)) / math_ops.log(2.0))), value.dtype)