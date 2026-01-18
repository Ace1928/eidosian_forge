import functools
import hashlib
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.util import tf_inspect
def _smallest_integer_by_dtype(dt):
    """Helper returning the smallest integer exactly representable by dtype."""
    if not _is_known_dtype(dt):
        raise TypeError('Unrecognized dtype: {}'.format(dt.name))
    if _is_known_unsigned_by_dtype(dt):
        return 0
    return -1 * _largest_integer_by_dtype(dt)