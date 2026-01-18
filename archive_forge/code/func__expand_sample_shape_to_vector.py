import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def _expand_sample_shape_to_vector(self, x, name):
    """Helper to `sample` which ensures input is 1D."""
    x_static_val = tensor_util.constant_value(x)
    if x_static_val is None:
        prod = math_ops.reduce_prod(x)
    else:
        prod = np.prod(x_static_val, dtype=x.dtype.as_numpy_dtype())
    ndims = x.get_shape().ndims
    if ndims is None:
        ndims = array_ops.rank(x)
        expanded_shape = util.pick_vector(math_ops.equal(ndims, 0), np.array([1], dtype=np.int32), array_ops.shape(x))
        x = array_ops.reshape(x, expanded_shape)
    elif ndims == 0:
        if x_static_val is not None:
            x = ops.convert_to_tensor(np.array([x_static_val], dtype=x.dtype.as_numpy_dtype()), name=name)
        else:
            x = array_ops.reshape(x, [1])
    elif ndims != 1:
        raise ValueError('Input is neither scalar nor vector.')
    return (x, prod)