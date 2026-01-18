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
def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    with self._name_scope(name, values=[sample_shape]):
        sample_shape = ops.convert_to_tensor(sample_shape, dtype=dtypes.int32, name='sample_shape')
        sample_shape, n = self._expand_sample_shape_to_vector(sample_shape, 'sample_shape')
        samples = self._sample_n(n, seed, **kwargs)
        batch_event_shape = array_ops.shape(samples)[1:]
        final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
        samples = array_ops.reshape(samples, final_shape)
        samples = self._set_sample_static_shape(samples, sample_shape)
        return samples