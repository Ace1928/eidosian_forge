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
def _truncate(self, new_rank: int) -> 'DynamicRaggedShape.Spec':
    """Truncate a ragged shape spec.

      For example, if the original spec s was for a shape:
      [3, [4, 1], 2, 7]

      Then truncate_dynamic_ragged_shape_spec(s, 3) is a spec for:
      [3, [4, 1], 2]

      Args:
        new_rank: the new rank

      Returns:
        A truncated DynamicRaggedShape.Spec.
      """
    if self.rank is None:
        return self._set_rank_if_unknown(new_rank)._truncate(new_rank)
    if new_rank == 0:
        return DynamicRaggedShape.Spec._from_tensor_shape([], 0, self.dtype)
    if new_rank == 1:
        vector_size = self._dimension(0)
        return DynamicRaggedShape.Spec._from_tensor_shape([vector_size], 0, self.dtype)
    if new_rank < self.num_row_partitions + 1:
        new_row_partitions = self._row_partitions[:new_rank - 1]
        new_static_inner_shape = tensor_shape.TensorShape([new_row_partitions[-1].nvals])
        return DynamicRaggedShape.Spec(row_partitions=new_row_partitions, static_inner_shape=new_static_inner_shape, dtype=self.dtype)
    else:
        remainder = new_rank - self.num_row_partitions
        new_static_inner_shape = self._static_inner_shape[:remainder]
        return DynamicRaggedShape.Spec(row_partitions=self._row_partitions, static_inner_shape=new_static_inner_shape, dtype=self.dtype)