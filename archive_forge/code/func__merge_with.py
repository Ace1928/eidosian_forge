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
def _merge_with(self, other: 'DynamicRaggedShape.Spec') -> 'DynamicRaggedShape.Spec':
    """Merges all information between two specs.

      Specs are expected to represent the same information modulo
      num_row_partitons.

      If the specs are of different ranks, then fail.

      Args:
        other: another Spec of the same rank.

      Returns:
        a Spec with the union of information.
      """
    max_num_row_partitions = max(self.num_row_partitions, other.num_row_partitions)
    a = self._with_num_row_partitions(max_num_row_partitions)
    b = other._with_num_row_partitions(max_num_row_partitions)
    new_rp = [a._merge_with(b) for a, b in zip(a._row_partitions, b._row_partitions)]
    new_static_inner_shape = a._static_inner_shape.merge_with(b._static_inner_shape)
    dtype = b.dtype if a.dtype == dtypes.int32 else dtypes.int64
    return DynamicRaggedShape.Spec(new_rp, new_static_inner_shape, dtype=dtype)