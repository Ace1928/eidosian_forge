import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _strip_padding(tensors, pad_len):
    """Strip the suffix padding added by _padded_split.

  Args:
    tensors: list of `tf.Tensor` of identical length 1D tensors.
    pad_len: number of elements to be stripped from the end of each tensor.

  Returns:
    list of `tf.Tensor` which are the stripped inputs.

  Raises:
    ValueError: tensors must be a non-empty list of 1D tensors, and
      each must be longer than pad_len.
  """
    if not tensors:
        raise ValueError('tensors cannot be empty')
    shape = tensors[0].shape
    if len(shape) > 1:
        raise ValueError('tensors must be 1D')
    prefix_len = int(shape[0] - pad_len)
    if prefix_len < 0:
        raise ValueError('pad_len longer than tensor')
    stripped = []
    for t in tensors:
        with ops.colocate_with(t):
            stripped.append(array_ops.slice(t, [0], [prefix_len]))
    return stripped