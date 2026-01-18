import typing
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _ragged_stack_concat_helper(rt_inputs, axis, stack_values):
    """Helper function to concatenate or stack ragged tensors.

  Args:
    rt_inputs: A list of RaggedTensors or Tensors to combine.
    axis: The axis along which to concatenate or stack.
    stack_values: A boolean -- if true, then stack values; otherwise,
      concatenate them.

  Returns:
    A RaggedTensor.
  Raises:
    ValueError: If rt_inputs is empty, or if axis is out of range.
  """
    if not rt_inputs:
        raise ValueError('rt_inputs may not be empty.')
    rt_inputs = [ragged_tensor.convert_to_tensor_or_ragged_tensor(rt_input, name='rt_input') for rt_input in rt_inputs]
    row_splits_dtype, rt_inputs = ragged_tensor.match_row_splits_dtypes(*rt_inputs, return_dtype=True)
    rt_inputs = list(rt_inputs)
    if len(rt_inputs) == 1 and (not stack_values):
        return rt_inputs[0]
    ndims = None
    for rt in rt_inputs:
        if ndims is None:
            ndims = rt.shape.ndims
        else:
            rt.shape.assert_has_rank(ndims)
    out_ndims = ndims if ndims is None or not stack_values else ndims + 1
    axis = array_ops.get_positive_axis(axis, out_ndims)
    if stack_values and ndims == 1 and (axis == 0):
        return ragged_tensor.RaggedTensor.from_row_lengths(values=array_ops.concat(rt_inputs, axis=0), row_lengths=array_ops.concat([array_ops.shape(r) for r in rt_inputs], axis=0))
    if all((not ragged_tensor.is_ragged(rt) for rt in rt_inputs)):
        if ndims is not None and (axis == out_ndims - 1 or axis == ndims - 1):
            if stack_values:
                return array_ops_stack.stack(rt_inputs, axis)
            else:
                return array_ops.concat(rt_inputs, axis)
    for i in range(len(rt_inputs)):
        if not ragged_tensor.is_ragged(rt_inputs[i]):
            rt_inputs[i] = ragged_tensor.RaggedTensor.from_tensor(rt_inputs[i], ragged_rank=1, row_splits_dtype=row_splits_dtype)
    ragged_rank = max(max((rt.ragged_rank for rt in rt_inputs)), 1)
    rt_inputs = [_increase_ragged_rank_to(rt, ragged_rank, row_splits_dtype) for rt in rt_inputs]
    if axis == 0:
        return _ragged_stack_concat_axis_0(rt_inputs, stack_values)
    elif axis == 1:
        return _ragged_stack_concat_axis_1(rt_inputs, stack_values)
    else:
        values = [rt.values for rt in rt_inputs]
        splits = [[rt_input.row_splits] for rt_input in rt_inputs]
        with ops.control_dependencies(ragged_util.assert_splits_match(splits)):
            return ragged_tensor.RaggedTensor.from_row_splits(_ragged_stack_concat_helper(values, axis - 1, stack_values), splits[0][0], validate=False)