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
def _add_fixed_len_sequence_feature(self, key, feature):
    """Adds a FixedLenSequenceFeature."""
    if not feature.dtype:
        raise ValueError(f'Missing type for feature {key}. Received feature={feature}.')
    if feature.shape is None:
        raise ValueError(f'Missing shape for feature {key}. Received feature={feature}.')
    self.dense_keys.append(key)
    self.dense_shapes.append(tensor_shape.as_shape(feature.shape))
    self.dense_types.append(feature.dtype)
    if feature.allow_missing:
        self.dense_defaults[key] = None
    if feature.default_value is not None:
        self.dense_defaults[key] = feature.default_value