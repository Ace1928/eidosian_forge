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
def _dimension(self, index: int) -> Optional[int]:
    """Get the size of dimension index, if known statically."""
    if index == 0:
        if self._row_partitions:
            return self._row_partitions[0].nrows
        elif self.inner_rank is None:
            return None
        elif self.inner_rank == 0:
            raise ValueError('Index out of range: 0.')
        else:
            return tensor_shape.dimension_value(self._static_inner_shape[0])
    if index <= len(self._row_partitions):
        return self._row_partitions[index - 1].uniform_row_length
    relative_index = index - self.num_row_partitions
    if self.inner_rank is None:
        return None
    elif self.inner_rank <= relative_index:
        raise ValueError(f'Index out of range: {index}.')
    else:
        return tensor_shape.dimension_value(self._static_inner_shape[relative_index])