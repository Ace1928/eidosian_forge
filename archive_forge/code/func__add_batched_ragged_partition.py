import collections
import re
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export
def _add_batched_ragged_partition(rt, partition, tensor_dict, feature_key, validate, outer_splits=None):
    """Adds a batched ragged partition tensor to a batched ragged tensor.

  Args:
    rt: A RaggedTensor with shape [batch_size, ...].
    partition: The partition configuration object.  Specifies the key that
      should be used to look up the partition tensor (unless partition is a
      RaggedFeature.UniformRowLength, in which case there is no partition
      tensor).  The specified tensor must have shape [batch_size, ...].
    tensor_dict: The dictionary mapping keys to tensors.
    feature_key: The name of the feature being parsed (for error messages).
    validate: Whether to validate that the values form a valid RaggedTensor.
    outer_splits: If not None, then we have two batch dimensions, and this
      is the row-splits for the collapsed batch dimension.  Every partition
      tensor must have an outer row_splits that matches this value.

  Returns:
    A new RaggedTensor where each batch item `rt[i]` has been partitioned
    using the `partition_t[i]`.
  """
    if isinstance(partition, RaggedFeature.UniformRowLength):
        if rt.ragged_rank > 1:
            length = ops.convert_to_tensor(partition.length, rt.row_splits.dtype)
            return ragged_tensor.RaggedTensor.from_row_splits(ragged_tensor.RaggedTensor.from_uniform_row_length(rt.values, length, validate=validate), rt.row_splits // length, validate=validate)
        else:
            reshaped_vals = array_ops.reshape(rt.values, array_ops.concat([[-1, partition.length], array_ops.shape(rt.values)[1:]], axis=0))
            return ragged_tensor.RaggedTensor.from_row_splits(reshaped_vals, rt.row_splits // partition.length, validate=validate)
    partition_t = tensor_dict[partition.key]
    if partition_t.values.dtype != rt.row_splits.dtype:
        partition_t = math_ops.cast(partition_t, rt.row_splits.dtype)
    checks = []
    if outer_splits is not None:
        if validate:
            checks.append(check_ops.assert_equal(outer_splits, partition_t.row_splits, message='Feature %s: values and partitions are not aligned' % feature_key))
        partition_t = partition_t.values
    with ops.control_dependencies(checks):
        if isinstance(partition, (RaggedFeature.RowSplits, RaggedFeature.RowLimits)):
            if isinstance(partition, RaggedFeature.RowSplits):
                partition_t = partition_t[:, 1:]
            adjusted_limits = partition_t.values + array_ops.repeat(rt.row_starts(), partition_t.row_lengths())
            return partition_t.with_values(ragged_tensor.RaggedTensor.from_row_limits(rt.values, adjusted_limits, validate=validate))
        elif isinstance(partition, RaggedFeature.RowStarts):
            adjusted_starts = partition_t.values + array_ops.repeat(rt.row_starts(), partition_t.row_lengths())
            return partition_t.with_values(ragged_tensor.RaggedTensor.from_row_starts(rt.values, adjusted_starts, validate=validate))
        elif isinstance(partition, RaggedFeature.RowLengths):
            return partition_t.with_values(ragged_tensor.RaggedTensor.from_row_lengths(rt.values, partition_t.values, validate=validate))
        elif isinstance(partition, RaggedFeature.ValueRowIds):
            nrows = math_ops.maximum(ragged_math_ops.reduce_max(partition_t + 1, axis=1), 0)
            adjusted_rowids = partition_t.values + array_ops.repeat(math_ops.cumsum(nrows, exclusive=True), partition_t.row_lengths())
            return ragged_tensor.RaggedTensor.from_row_lengths(ragged_tensor.RaggedTensor.from_value_rowids(rt.values, adjusted_rowids, validate=validate), nrows, validate=validate)
        raise ValueError(f'Unhandled partition type {partition!r}')