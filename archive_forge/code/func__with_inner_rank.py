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
def _with_inner_rank(self, inner_rank):
    """Returns the same shape but a different inner_rank.

    All dimensions that are to be represented in the inner_shape must be dense.
    See inner_rank.

    Args:
      inner_rank: the new inner_rank of the shape.

    Returns:
      the same shape but a different inner_rank

    Raises:
      ValueError if the new dense rank is invalid, or the old rank is unknown.
    """
    rank = self.rank
    if rank is None:
        raise ValueError('Rank must be known to adjust inner_rank')
    elif rank < 2:
        if inner_rank == rank:
            return self
        raise ValueError('Cannot change inner_rank if rank < 2')
    else:
        new_num_row_partitions = rank - inner_rank
        return self._with_num_row_partitions(new_num_row_partitions)