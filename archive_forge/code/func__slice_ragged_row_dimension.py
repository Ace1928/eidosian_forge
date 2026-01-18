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
def _slice_ragged_row_dimension(rt_input, row_key):
    """Slice the outer dimension of `rt_input` according to the given `slice`.

  Args:
    rt_input: The `RaggedTensor` to slice.
    row_key: The `slice` object that should be used to slice `rt_input`.

  Returns:
    A `RaggedTensor` containing the indicated slice of `rt_input`.
  """
    if row_key.start is None and row_key.stop is None and (row_key.step is None):
        return rt_input
    new_starts = rt_input.row_splits[:-1][row_key]
    new_limits = rt_input.row_splits[1:][row_key]
    zero_pad = array_ops.zeros([1], rt_input.row_splits.dtype)
    if row_key.step is None or row_key.step == 1:
        new_splits = array_ops.concat([zero_pad[array_ops.size(new_starts):], new_starts[:1], new_limits], axis=0)
        values_start = new_splits[0]
        values_limit = new_splits[-1]
        return ragged_tensor.RaggedTensor.from_row_splits(rt_input.values[values_start:values_limit], new_splits - values_start, validate=False)
    else:
        return _build_ragged_tensor_from_value_ranges(new_starts, new_limits, 1, rt_input.values)