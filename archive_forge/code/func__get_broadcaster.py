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
def _get_broadcaster(source_shape, target_shape):
    """Get a _Broadcaster from source_shape to target_shape."""
    if source_shape.dtype != target_shape.dtype:
        raise ValueError('The source and target row_split dtypes should be equal')
    if source_shape.rank is None or target_shape.rank is None:
        raise ValueError('Rank of source and target must be statically known')
    elif source_shape.rank > target_shape.rank:
        raise ValueError('Cannot broadcast to a shape with smaller rank')
    elif source_shape.rank == 0:
        return _Broadcaster(source_shape, target_shape, [])
    elif target_shape.rank == 1:
        assert source_shape.rank == 1
        layer = _LayerBroadcaster.first_layer(source_shape.inner_shape[0], target_shape.inner_shape[0])
        return _Broadcaster(source_shape, target_shape, [layer])
    assert source_shape.rank <= target_shape.rank
    assert target_shape.rank >= 2
    assert source_shape.rank >= 1
    source_rps = source_shape._as_row_partitions()
    target_rps = target_shape._as_row_partitions()
    assert len(target_rps) >= 1
    assert len(source_rps) <= len(target_rps)
    source_nrows = source_shape[0]
    if len(source_rps) < len(target_rps):
        neg_one_source_rp = RowPartition.from_uniform_row_length(uniform_row_length=source_nrows, nrows=1, nvals=source_nrows)
        neg_one_target_rp = target_rps[-(len(source_rps) + 1)]
        neg_one_broadcaster = _LayerBroadcaster.get_singleton_broadcaster(neg_one_target_rp.nrows())
        zeroth_broadcaster = neg_one_broadcaster.next_layer(neg_one_source_rp, neg_one_target_rp)
        target_rps_tail = target_rps[-len(source_rps):] if len(source_rps) >= 1 else []
        layers = _get_layer_broadcasters_from_rps(zeroth_broadcaster, source_rps, target_rps_tail)
        return _Broadcaster(source_shape, target_shape, layers)
    else:
        assert len(target_rps) == len(source_rps)
        zeroth_broadcaster = _LayerBroadcaster.first_layer(source_rps[0].nrows(), target_rps[0].nrows())
        layers = _get_layer_broadcasters_from_rps(zeroth_broadcaster, source_rps, target_rps)
        return _Broadcaster(source_shape, target_shape, layers)