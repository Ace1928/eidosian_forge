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
def _broadcast_dynamic_shape_from_rps(a_zero: _LayerBroadcaster, b_zero: _LayerBroadcaster, a_rps: Sequence[RowPartition], b_rps: Sequence[RowPartition]) -> Tuple[Sequence[RowPartition], Sequence[_LayerBroadcaster], Sequence[_LayerBroadcaster]]:
    """Create BroadcastLayers from two shapes to a target shape.


      *--a_zero->*<-b_zero-*
      |          |         |
   a_rps[0]    c_rps[0]  b_rps[0]
      |          |         |
      V          V         V
      *--ac[1]-->*<-bc[1]--*
      |          |         |
   a_rps[1]   c_rps[0]   b_rps[1]
      |          |         |
      V          V         V
      *--ac[2]-->*<-bc[2]--*

  Note: ac[0]=a_zero, and bc[0]=b_zero.
  Args:
    a_zero: broadcaster from rows of a_rps[0] to target shape.
    b_zero: broadcaster from rows of b_rps[0] to target shape.
    a_rps: RowPartitions of first shape.
    b_rps: RowPartitions of second shape, equal in length to a_rps.

  Returns:
    (c_rps, ac, bc) where:
    c_rps: RowPartitions of target shape.
    ac: layers broadcasting from the first shape.
    bc: layers broadcasting from the second shape.
  """
    assert len(a_rps) == len(b_rps)
    if a_rps:
        c_1, ac_1, bc_1 = _broadcast_dynamic_shape_next_layer(a_zero, b_zero, a_rps[0], b_rps[0])
        c_suffix, a_layers, b_layers = _broadcast_dynamic_shape_from_rps(ac_1, bc_1, a_rps[1:], b_rps[1:])
        return ([c_1] + c_suffix, [ac_1] + a_layers, [bc_1] + b_layers)
    else:
        return ([], [], [])