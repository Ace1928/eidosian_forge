from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _broadcast_cat_event_and_params(event, params, base_dtype):
    """Broadcasts the event or distribution parameters."""
    if event.dtype.is_integer:
        pass
    elif event.dtype.is_floating:
        event = math_ops.cast(event, dtype=dtypes.int32)
    else:
        raise TypeError('`value` should have integer `dtype` or `self.dtype` ({})'.format(base_dtype))
    shape_known_statically = params.shape.ndims is not None and params.shape[:-1].is_fully_defined() and event.shape.is_fully_defined()
    if not shape_known_statically or params.shape[:-1] != event.shape:
        params *= array_ops.ones_like(event[..., array_ops.newaxis], dtype=params.dtype)
        params_shape = array_ops.shape(params)[:-1]
        event *= array_ops.ones(params_shape, dtype=event.dtype)
        if params.shape.ndims is not None:
            event.set_shape(tensor_shape.TensorShape(params.shape[:-1]))
    return (event, params)