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
def _alt_inner_shape(self, new_inner_rank):
    """Get an alternative inner shape with higher or lower rank.

    For the rank of the inner shape to be be higher, the last few ragged
    dimensions must have uniform_row_length.

    Args:
      new_inner_rank: the new rank of the inner_shape

    Returns:
       A new inner_shape of rank new_inner_rank.
    """
    if new_inner_rank == 0:
        raise ValueError('new_inner_rank cannot be zero')
    elif self.inner_rank == 0:
        raise ValueError('old inner_rank cannot be zero')
    elif new_inner_rank == self.inner_rank:
        return self.inner_shape
    elif new_inner_rank < self.inner_rank:
        if self._static_inner_shape.is_fully_defined():
            return _alt_inner_shape_from_tensor_shape(self._static_inner_shape, self.dtype, new_inner_rank)
        first_dimension = self._num_slices_in_dimension(-new_inner_rank)
        if new_inner_rank == 1:
            return array_ops.expand_dims(first_dimension, 0)
        remaining_dimensions = self.inner_shape[1 - new_inner_rank:]
        return array_ops.concat([array_ops.expand_dims(first_dimension, 0), remaining_dimensions], axis=0)
    else:
        assert new_inner_rank > self.inner_rank
        new_dimensions = new_inner_rank - self.inner_rank
        if any([not x.is_uniform() for x in self.row_partitions[-new_dimensions:]]):
            raise ValueError('Cannot get an inner shape over a ragged dimension')
        first_dimension = self._num_slices_in_dimension(-new_inner_rank)
        new_dimensions = new_inner_rank - self.inner_rank
        new_dims = [first_dimension] + [x.uniform_row_length() for x in self.row_partitions[-new_dimensions:]]
        return array_ops.concat([array_ops_stack.stack(new_dims), self.inner_shape[1:]], axis=0)