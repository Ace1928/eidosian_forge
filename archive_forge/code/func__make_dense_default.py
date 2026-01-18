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
def _make_dense_default(self, key, shape, dtype):
    """Construct the default value tensor for a specified dense feature.

    Args:
      key: The key string identifying the dense feature.
      shape: The dense feature's shape.
      dtype: The dense feature's dtype.

    Returns:
      A Tensor.
    """
    default_value = self.dense_defaults.get(key)
    if shape.ndims is not None and shape.ndims > 0 and (shape.dims[0].value is None):
        if default_value is None:
            default_value = ops.convert_to_tensor('' if dtype == dtypes.string else 0, dtype=dtype)
        else:
            key_name = 'padding_' + re.sub('[^A-Za-z0-9_.\\-/]', '_', key)
            default_value = ops.convert_to_tensor(default_value, dtype=dtype, name=key_name)
            default_value = array_ops.reshape(default_value, [])
    elif default_value is None:
        default_value = constant_op.constant([], dtype=dtype)
    elif not isinstance(default_value, tensor.Tensor):
        key_name = 'key_' + re.sub('[^A-Za-z0-9_.\\-/]', '_', key)
        default_value = ops.convert_to_tensor(default_value, dtype=dtype, name=key_name)
        default_value = array_ops.reshape(default_value, shape)
    return default_value