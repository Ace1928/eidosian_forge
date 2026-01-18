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
def _add_sparse_feature(self, key, feature):
    """Adds a SparseFeature."""
    if not feature.index_key:
        raise ValueError(f'Missing index_key for SparseFeature {feature}.')
    if not feature.value_key:
        raise ValueError(f'Missing value_key for SparseFeature {feature}.')
    if not feature.dtype:
        raise ValueError(f'Missing type for feature {key}. Received feature={feature}.')
    index_keys = feature.index_key
    if isinstance(index_keys, str):
        index_keys = [index_keys]
    elif len(index_keys) > 1:
        tf_logging.warning('SparseFeature is a complicated feature config and should only be used after careful consideration of VarLenFeature.')
    for index_key in sorted(index_keys):
        self._add_sparse_key(index_key, dtypes.int64)
    self._add_sparse_key(feature.value_key, feature.dtype)