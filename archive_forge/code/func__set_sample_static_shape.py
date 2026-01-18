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
def _set_sample_static_shape(self, x, sample_shape):
    """Helper to `sample`; sets static shape info."""
    sample_shape = tensor_shape.TensorShape(tensor_util.constant_value(sample_shape))
    ndims = x.get_shape().ndims
    sample_ndims = sample_shape.ndims
    batch_ndims = self.batch_shape.ndims
    event_ndims = self.event_shape.ndims
    if ndims is None and sample_ndims is not None and (batch_ndims is not None) and (event_ndims is not None):
        ndims = sample_ndims + batch_ndims + event_ndims
        x.set_shape([None] * ndims)
    if ndims is not None and sample_ndims is not None:
        shape = sample_shape.concatenate([None] * (ndims - sample_ndims))
        x.set_shape(x.get_shape().merge_with(shape))
    if ndims is not None and event_ndims is not None:
        shape = tensor_shape.TensorShape([None] * (ndims - event_ndims)).concatenate(self.event_shape)
        x.set_shape(x.get_shape().merge_with(shape))
    if batch_ndims is not None:
        if ndims is not None:
            if sample_ndims is None and event_ndims is not None:
                sample_ndims = ndims - batch_ndims - event_ndims
            elif event_ndims is None and sample_ndims is not None:
                event_ndims = ndims - batch_ndims - sample_ndims
        if sample_ndims is not None and event_ndims is not None:
            shape = tensor_shape.TensorShape([None] * sample_ndims).concatenate(self.batch_shape).concatenate([None] * event_ndims)
            x.set_shape(x.get_shape().merge_with(shape))
    return x