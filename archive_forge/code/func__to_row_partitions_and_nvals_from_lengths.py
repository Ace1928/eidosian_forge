import abc
from typing import Any, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.ops.ragged.row_partition import RowPartitionSpec
from tensorflow.python.types import core
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def _to_row_partitions_and_nvals_from_lengths(lengths: Sequence[Union[int, Sequence[int]]], dtype=None) -> Tuple[Sequence[RowPartition], int]:
    """Allow ragged and uniform shapes to be specified.

  For example, [2, [2,1], 2] represents a shape like:
  [[[0, 0], [0, 0]], [[0, 0]]]

  Args:
    lengths: a list of integers and lists of integers.
    dtype: dtype of the shape (tf.int32 or tf.int64)

  Returns:
    a sequence of RowPartitions, and the number of values of the last partition.
  """
    size_so_far = lengths[0]
    result = []
    for current_lengths in lengths[1:]:
        if isinstance(current_lengths, int):
            nrows = size_so_far
            nvals = current_lengths * nrows
            size_so_far = nvals
            result.append(RowPartition.from_uniform_row_length(current_lengths, nvals, nrows=nrows, dtype_hint=dtype))
        else:
            if size_so_far != len(current_lengths):
                raise ValueError('Shape not consistent.')
            result.append(RowPartition.from_row_lengths(current_lengths, dtype_hint=dtype))
            size_so_far = sum(current_lengths)
    return (result, size_so_far)