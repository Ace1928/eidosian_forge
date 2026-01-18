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
def _build_ragged_tensor_from_value_ranges(starts, limits, step, values):
    """Returns a `RaggedTensor` containing the specified sequences of values.

  Returns a RaggedTensor `output` where:

  ```python
  output.shape[0] = starts.shape[0]
  output[i] = values[starts[i]:limits[i]:step]
  ```

  Requires that `starts.shape == limits.shape` and
  `0 <= starts[i] <= limits[i] <= values.shape[0]`.

  Args:
    starts: 1D integer Tensor specifying the start indices for the sequences of
      values to include.
    limits: 1D integer Tensor specifying the limit indices for the sequences of
      values to include.
    step: Integer value specifying the step size for strided slices.
    values: The set of values to select from.

  Returns:
    A `RaggedTensor`.

  Raises:
    ValueError: Until the prerequisite ops are checked in.
  """
    if step is None:
        step = 1
    step = ops.convert_to_tensor(step, name='step')
    if step.dtype.is_integer:
        step = math_ops.cast(step, starts.dtype)
    else:
        raise TypeError('slice strides must be integers or None')
    value_indices = ragged_math_ops.range(starts, limits, step, row_splits_dtype=starts.dtype)
    if isinstance(values, ragged_tensor.RaggedTensor):
        gathered_values = ragged_gather_ops.gather(params=values, indices=value_indices.values)
    else:
        gathered_values = array_ops.gather(params=values, indices=value_indices.values)
    return value_indices.with_values(gathered_values)