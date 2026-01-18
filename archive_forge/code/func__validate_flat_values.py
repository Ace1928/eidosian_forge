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
def _validate_flat_values(self, flat_values):
    """Test if flat_values have the right nvals."""
    if not isinstance(flat_values, tensor_lib.Tensor):
        return flat_values
    if self.row_partitions:
        last_row_partition = self.row_partitions[-1]
        flat_values_shape = flat_values.shape
        if flat_values_shape is None:
            return self._validate_flat_values_dynamically(flat_values)
        first_dim_flat_values = flat_values_shape[0]
        if isinstance(first_dim_flat_values, tensor_shape.Dimension):
            first_dim_flat_values = first_dim_flat_values.value
        if first_dim_flat_values is None:
            return self._validate_flat_values_dynamically(flat_values)
        static_nvals = last_row_partition.static_nvals
        if static_nvals is None:
            return self._validate_flat_values_dynamically(flat_values)
        if first_dim_flat_values != static_nvals:
            raise ValueError('Last row partition does not match flat_values.')
    return flat_values