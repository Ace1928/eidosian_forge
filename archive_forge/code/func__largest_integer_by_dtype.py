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
def _largest_integer_by_dtype(dt):
    """Helper returning the largest integer exactly representable by dtype."""
    if not _is_known_dtype(dt):
        raise TypeError('Unrecognized dtype: {}'.format(dt.name))
    if dt.is_floating:
        return int(2 ** (np.finfo(dt.as_numpy_dtype).nmant + 1))
    if dt.is_integer:
        return np.iinfo(dt.as_numpy_dtype).max
    if dt.base_dtype == dtypes.bool:
        return int(1)
    raise TypeError('Unrecognized dtype: {}'.format(dt.name))