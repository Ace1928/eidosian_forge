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
def _ragged_getitem(rt_input, key_list):
    """Helper for indexing and slicing ragged tensors with __getitem__().

  Extracts the specified piece of the `rt_input`.  See
  `RaggedTensor.__getitem__` for examples and restrictions.

  Args:
    rt_input: The `RaggedTensor` from which a piece should be returned.
    key_list: The list of keys specifying which piece to return. Each key
      corresponds with a separate dimension.

  Returns:
    The indicated piece of rt_input.

  Raises:
    ValueError: If `key_list` is not supported.
    TypeError: If any keys in `key_list` have an unsupported type.
  """
    if not key_list:
        return rt_input
    row_key = key_list[0]
    inner_keys = key_list[1:]
    if row_key is Ellipsis:
        expanded_key_list = _expand_ellipsis(key_list, rt_input.shape.ndims)
        return _ragged_getitem(rt_input, expanded_key_list)
    if row_key is array_ops.newaxis:
        inner_rt = _ragged_getitem(rt_input, inner_keys)
        nsplits = tensor_shape.dimension_at_index(inner_rt.row_splits.shape, 0)
        if nsplits.value is not None:
            nsplits = nsplits.value
        else:
            nsplits = array_ops.shape(inner_rt.row_splits, out_type=inner_rt.row_splits.dtype)[0]
        return ragged_tensor.RaggedTensor.from_uniform_row_length(inner_rt, nsplits - 1, nrows=1, validate=False)
    if isinstance(row_key, slice):
        sliced_rt_input = _slice_ragged_row_dimension(rt_input, row_key)
        if rt_input.uniform_row_length is not None:
            sliced_rt_input = ragged_tensor.RaggedTensor.from_uniform_row_length(sliced_rt_input.values, rt_input.uniform_row_length, nrows=sliced_rt_input.nrows())
        return _ragged_getitem_inner_dimensions(sliced_rt_input, inner_keys)
    else:
        starts = rt_input.row_splits[:-1]
        limits = rt_input.row_splits[1:]
        if context.executing_eagerly():
            try:
                if int(row_key) >= len(starts):
                    raise IndexError('Row key {} out of bounds'.format(row_key))
            except (TypeError, ValueError):
                pass
        row = rt_input.values[starts[row_key]:limits[row_key]]
        return row.__getitem__(inner_keys)