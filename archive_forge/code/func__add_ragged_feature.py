import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _add_ragged_feature(self, key, feature):
    """Adds a RaggedFeature."""
    value_key = key if feature.value_key is None else feature.value_key
    self._add_ragged_key(value_key, feature.dtype, feature.row_splits_dtype)
    for partition in feature.partitions:
        if not isinstance(partition, RaggedFeature.UniformRowLength):
            self._add_ragged_key(partition.key, dtypes.int64, feature.row_splits_dtype)