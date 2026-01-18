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
def _fix_start_index(index, rank, num_row_partitions):
    """Slice indexes are always silently truncated."""
    if index < 0:
        if rank is None:
            raise ValueError('Rank must be known to use __getitem__ on a negative index.')
        index = rank + index
    if index < 0:
        index = 0
    if num_row_partitions > 0 and index <= num_row_partitions + 1:
        return index
    if index == 0:
        return index
    if rank is None:
        raise ValueError('Rank must be known to use __getitem__ on a large index.')
    if index >= rank:
        index = rank
    return index