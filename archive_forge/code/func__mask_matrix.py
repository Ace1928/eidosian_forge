import re
import numpy as np
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _mask_matrix(length):
    """Computes t_n = exp(sqrt(-1) * pi * n^2 / line_len)."""
    a = _array_ops.tile(_array_ops.expand_dims(_math_ops.range(length), 0), (length, 1))
    b = _array_ops.transpose(a, [1, 0])
    return _math_ops.exp(-2j * np.pi * _math_ops.cast(a * b, complex_dtype) / _math_ops.cast(length, complex_dtype))