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
def _expand_ellipsis(key_list, num_remaining_dims):
    """Expands the ellipsis at the start of `key_list`.

  Assumes that the first element of `key_list` is Ellipsis.  This will either
  remove the Ellipsis (if it corresponds to zero indices) or prepend a new
  `slice(None, None, None)` (if it corresponds to more than zero indices).

  Args:
    key_list: The arguments to `__getitem__()`.
    num_remaining_dims: The number of dimensions remaining.

  Returns:
    A copy of `key_list` with he ellipsis expanded.
  Raises:
    ValueError: If ragged_rank.shape.ndims is None
    IndexError: If there are too many elements in `key_list`.
  """
    if num_remaining_dims is None:
        raise ValueError('Ellipsis not supported for unknown shape RaggedTensors')
    num_indices = sum((1 for idx in key_list if idx is not array_ops.newaxis))
    if num_indices > num_remaining_dims + 1:
        raise IndexError('Too many indices for RaggedTensor')
    elif num_indices == num_remaining_dims + 1:
        return key_list[1:]
    else:
        return [slice(None, None, None)] + key_list