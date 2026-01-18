import collections
import math
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
def _flatten_tensors(tensors):
    """Check tensors for isomorphism and flatten.

  Args:
    tensors: list of `tf.Tensor` which must all have the same shape.

  Returns:
    tensors: a list of `tf.Tensor` which are flattened (1D) views of tensors
    shape: the original shape of each element of input tensors

  Raises:
    ValueError: tensors are empty or non-isomorphic or have unknown shape.
  """
    if not tensors:
        raise ValueError('tensors cannot be empty')
    shape = tensors[0].shape
    for tensor in tensors:
        shape = shape.merge_with(tensor.shape)
    if not shape.is_fully_defined():
        raise ValueError('Tensors must have statically known shape.')
    if len(shape) != 1:
        reshaped = []
        for t in tensors:
            with ops.colocate_with(t):
                reshaped.append(array_ops.reshape(t, [-1]))
        tensors = reshaped
    return (tensors, shape)