from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
def _inplace_helper(x, i, v, op):
    """Applies an inplace op on (x, i, v).

  op is one of gen_array_ops.alias_inplace_update,
  gen_array_ops.alias_inplace_add, or gen_array_ops.alias_inplace_sub.

  If i is None, x and v must be the same shape. Computes
    x op v;
  If i is a scalar, x has a rank 1 higher than v's. Computes
    x[i, :] op v;
  Otherwise, x and v must have the same rank. Computes
    x[i, :] op v;

  Args:
    x: A Tensor.
    i: None, a scalar or a vector.
    v: A Tensor.
    op: alias_inplace_update, alias_inplace_add, or alias_inplace_sub.

  Returns:
    Returns x.

  """
    x = ops.convert_to_tensor(x)
    v = ops.convert_to_tensor(v, x.dtype)
    if i is None:
        return array_ops.reshape(op(array_ops.reshape(x, [1, -1]), [0], array_ops.reshape(v, [1, -1])), array_ops.shape(x))
    i = math_ops.cast(i, dtypes.int32)
    if i.get_shape().ndims == 0:
        return op(x, array_ops.reshape(i, [1]), array_ops.expand_dims(v, 0))
    return op(x, i, v)