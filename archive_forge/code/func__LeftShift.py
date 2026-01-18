from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as _linalg
def _LeftShift(x):
    """Shifts next-to-last dimension to the left, adding zero on the right."""
    rank = array_ops.rank(x)
    zeros = array_ops.zeros((rank - 2, 2), dtype=dtypes.int32)
    pad = array_ops.concat([zeros, array_ops.constant([[0, 1], [0, 0]])], axis=0)
    return array_ops.pad(x[..., 1:, :], pad)