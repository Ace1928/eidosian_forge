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
def _is_scalar_helper(self, static_shape, dynamic_shape_fn):
    """Implementation for `is_scalar_batch` and `is_scalar_event`."""
    if static_shape.ndims is not None:
        return static_shape.ndims == 0
    shape = dynamic_shape_fn()
    if shape.get_shape().ndims is not None and shape.get_shape().dims[0].value is not None:
        return shape.get_shape().as_list() == [0]
    return math_ops.equal(array_ops.shape(shape)[0], 0)