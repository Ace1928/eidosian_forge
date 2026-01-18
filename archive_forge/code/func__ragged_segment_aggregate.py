import functools
import typing
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _ragged_segment_aggregate(unsorted_segment_op, data, segment_ids, num_segments, separator=None, name=None):
    """Aggregates along segments of a RaggedTensor using `unsorted_segment_op`.

  Returns a RaggedTensor `output` with `num_segments` rows, where the row
  `output[i]` is formed by combining all rows of `data` whose corresponding
  `segment_id` is `i`.  The values in each row are combined using
  `unsorted_segment_op`.

  The length of the row `output[i]` will be the maximum of the lengths of
  all rows of `data` whose corresponding `segment_id` is `i`.  If no `data`
  rows correspond to a given segment ID, then the output row for that segment
  ID will be empty.

  Args:
    unsorted_segment_op: The tensorflow `op` that should be used to combine
      values in each row.  Must have the same signature and basic behavior as
      `unsorted_segment_sum`, `unsorted_segment_max`, etc.
    data: A `RaggedTensor` containing the values to be combined.
    segment_ids: A `Tensor` or `RaggedTensor`.  Must have type `int64` or
      `int32`.  `segment_ids.shape` must be a prefix of `data.shape`.
      `segment_ids` is not required to be sorted.
    num_segments: An `int32` or `int64` scalar.
    separator: An optional string. Defaults to None. The separator to use when
      joining. Only used for string types.
    name: A name prefix for the returned tensor (optional).

  Returns:
    A `RaggedTensor` containing the aggregated values.  The returned tensor
    has the same dtype as `data`, and its shape is
    `[num_segments] + data.shape[segment_ids.rank:]`.
  Raises:
    ValueError: If segment_ids.shape is not a prefix of data.shape.
  """
    if not (ragged_tensor.is_ragged(data) or ragged_tensor.is_ragged(segment_ids)):
        if separator is not None:
            return unsorted_segment_op(data, segment_ids, num_segments, separator, name)
        else:
            return unsorted_segment_op(data, segment_ids, num_segments, name)
    with ops.name_scope(name, 'RaggedSegment', [data, segment_ids, num_segments]) as name:
        data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name='data')
        segment_ids = ragged_tensor.convert_to_tensor_or_ragged_tensor(segment_ids, name='segment_ids')
        data, segment_ids = ragged_tensor.match_row_splits_dtypes(data, segment_ids)
        if segment_ids.dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('segment_ids must have dtype int32 or int64.')
        if ragged_tensor.is_ragged(segment_ids):
            if not ragged_tensor.is_ragged(data):
                raise ValueError('segment_ids.shape must be a prefix of data.shape, but segment_ids is ragged and data is not.')
            check_splits = check_ops.assert_equal(segment_ids.row_splits, data.row_splits, message='segment_ids.shape must be a prefix of data.shape')
            with ops.control_dependencies([check_splits]):
                return _ragged_segment_aggregate(unsorted_segment_op, data.values, segment_ids.values, num_segments, separator)
        data_row_lengths = data.row_splits[1:] - data.row_splits[:-1]
        output_row_lengths = math_ops.maximum(math_ops.unsorted_segment_max(data_row_lengths, segment_ids, num_segments), 0)
        output_splits = array_ops.concat([array_ops.zeros([1], output_row_lengths.dtype), math_ops.cumsum(output_row_lengths)], axis=0)
        data_row_to_out_row_start = array_ops.gather(output_splits, segment_ids)
        data_row_to_out_row_limit = data_row_to_out_row_start + data_row_lengths
        data_val_to_out_val_index = range(data_row_to_out_row_start, data_row_to_out_row_limit).values
        output_values = _ragged_segment_aggregate(unsorted_segment_op, data.values, data_val_to_out_val_index, output_splits[-1], separator)
        return ragged_tensor.RaggedTensor.from_row_splits(output_values, output_splits, validate=False)