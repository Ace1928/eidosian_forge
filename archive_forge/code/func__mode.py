import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@distribution_util.AppendDocstring('Note: The mode is undefined when `concentration1 <= 1` or\n      `concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`\n      is used for undefined modes. If `self.allow_nan_stats` is `False` an\n      exception is raised when one or more modes are undefined.')
def _mode(self):
    mode = (self.concentration1 - 1.0) / (self.total_concentration - 2.0)
    if self.allow_nan_stats:
        nan = array_ops.fill(self.batch_shape_tensor(), np.array(np.nan, dtype=self.dtype.as_numpy_dtype()), name='nan')
        is_defined = math_ops.logical_and(self.concentration1 > 1.0, self.concentration0 > 1.0)
        return array_ops.where_v2(is_defined, mode, nan)
    return control_flow_ops.with_dependencies([check_ops.assert_less(array_ops.ones([], dtype=self.dtype), self.concentration1, message='Mode undefined for concentration1 <= 1.'), check_ops.assert_less(array_ops.ones([], dtype=self.dtype), self.concentration0, message='Mode undefined for concentration0 <= 1.')], mode)