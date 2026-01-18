from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _if_ge_zero(value, true_fn, false_fn):
    """Returns `true_fn() if value >= 0 else false_fn()`."""
    if isinstance(value, tensor_lib.Tensor):
        const_value = tensor_util.constant_value(value)
        if const_value is None:
            return cond.cond(value >= 0, true_fn, false_fn)
        else:
            value = const_value
    if value >= 0:
        return true_fn()
    else:
        return false_fn()