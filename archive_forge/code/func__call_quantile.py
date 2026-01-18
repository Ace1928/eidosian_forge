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
def _call_quantile(self, value, name, **kwargs):
    with self._name_scope(name, values=[value]):
        value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
        return self._quantile(value, **kwargs)