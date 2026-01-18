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
def _gather(params, indices, axis, batch_dims):
    """Helper that implements the body for ragged gather().

  Assumes that `params` and `indices` have been converted to tensors or
  ragged tensors, and that `axis` and `batch_dims` have been normalized to
  be positive.  (So these conversions & normalizations can be skipped in
  recursive calls to _gather).

  Args:
    params: The tensor from which to gather values.
    indices: The indices of values to gather.
    axis: The axis in `params` to gather `indices` from.
    batch_dims: The number of batch dimensions.

  Returns:
    A potentially ragged tensor.
  """
    params_is_ragged = ragged_tensor.is_ragged(params)
    indices_is_ragged = ragged_tensor.is_ragged(indices)
    if not (params_is_ragged or indices_is_ragged):
        return array_ops.gather(params, indices, axis=axis, batch_dims=batch_dims)
    if batch_dims > 0:
        return _batch_gather(params, indices, axis, batch_dims)
    if axis > 0:
        return _axis_gather(params, indices, axis)
    if indices_is_ragged:
        return indices.with_values(_gather(params, indices.values, 0, 0))
    if indices.shape.ndims is None:
        raise ValueError('rank(indices) must be known statically')
    out_ragged_rank = indices.shape.ndims + len(params.nested_row_splits) - 1
    result = gen_ragged_array_ops.ragged_gather(indices=indices, params_dense_values=params.flat_values, params_nested_splits=params.nested_row_splits, OUTPUT_RAGGED_RANK=out_ragged_rank)
    result = ragged_tensor.RaggedTensor.from_nested_row_splits(result.output_dense_values, result.output_nested_splits, validate=False)
    if indices.shape.ndims > 1:
        target = result
        indices_shape = array_ops.shape(indices, out_type=params.row_splits.dtype)
        shape_cumprod = math_ops.cumprod(indices_shape)
        for dim in range(indices.shape.ndims - 1):
            target._cached_nrows = shape_cumprod[dim]
            target._uniform_row_length = indices_shape[dim + 1]
            target = target.values
    return result