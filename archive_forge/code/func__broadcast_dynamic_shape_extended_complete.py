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
def _broadcast_dynamic_shape_extended_complete(a: DynamicRaggedShape, b: DynamicRaggedShape, b_rps: Sequence[RowPartition], c_suffix: Sequence[RowPartition], ac: Sequence[_LayerBroadcaster], bc_suffix: Sequence[_LayerBroadcaster]) -> Tuple[DynamicRaggedShape, _Broadcaster, _Broadcaster]:
    """Helper for broadcast_dynamic_shape_extended."""
    c_prefix = b_rps[:-len(c_suffix)]
    bc_prefix_length = b.rank - len(bc_suffix)
    bc_prefix = [_LayerBroadcaster.get_identity_broadcaster(b._num_slices_in_dimension(i)) for i in range(bc_prefix_length)]
    c_num_row_partitions = _get_broadcast_num_row_partitions(a, b)
    c_raw = DynamicRaggedShape.from_row_partitions(c_prefix + tuple(c_suffix))
    c = c_raw._with_num_row_partitions(c_num_row_partitions)
    return (c, _Broadcaster(a, c, ac), _Broadcaster(b, c, bc_prefix + bc_suffix))