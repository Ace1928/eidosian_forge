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
def _set_rank_if_unknown(self, new_rank: int) -> 'DynamicRaggedShape.Spec':
    """Ensures this has a known rank at least new_rank."""
    if new_rank is None:
        raise TypeError('new_rank is None, but expected int')
    if new_rank < 0:
        raise ValueError('Rank must be non-negative')
    current_rank = self.rank
    if current_rank is not None and current_rank < new_rank:
        raise ValueError('Rank is {current_rank}, expected at least {new_rank}.'.format(current_rank=current_rank, new_rank=new_rank))
    if current_rank is not None:
        return self
    if self._row_partitions:
        new_inner_rank = max(new_rank - self.num_row_partitions, 1)
        first_dim = self._row_partitions[-1].nvals
        static_inner_shape = tensor_shape.TensorShape([first_dim] + [None] * (new_inner_rank - 1))
    else:
        static_inner_shape = tensor_shape.TensorShape([None] * new_rank)
    return DynamicRaggedShape.Spec(row_partitions=self._row_partitions, static_inner_shape=static_inner_shape, dtype=self.dtype)