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
def _get_broadcast_num_row_partitions(a: DynamicRaggedShape, b: DynamicRaggedShape):
    """Returns broadcast_dynamic_shape(a, b).num_row_partitions."""
    if a.num_row_partitions == 0 and b.num_row_partitions == 0:
        return 0
    expanded_num_row_partitions_a = a.num_row_partitions + max(0, b.rank - a.rank)
    expanded_num_row_partitions_b = b.num_row_partitions + max(0, a.rank - b.rank)
    if a.num_row_partitions == 0:
        return expanded_num_row_partitions_b
    if b.num_row_partitions == 0:
        return expanded_num_row_partitions_a
    return max(expanded_num_row_partitions_a, expanded_num_row_partitions_b)