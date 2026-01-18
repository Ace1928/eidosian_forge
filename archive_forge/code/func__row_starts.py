from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
def _row_starts(t, dtype):
    """Returns the start indices for the rows in `t`."""
    if isinstance(t, ragged_tensor.RaggedTensor):
        return math_ops.cast(t.row_starts(), dtype)
    else:
        t_shape = array_ops.shape(t, out_type=dtype)
        return math_ops.range(t_shape[0]) * t_shape[1]