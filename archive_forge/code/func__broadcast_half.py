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
def _broadcast_half(ac_0: _LayerBroadcaster, a_1: RowPartition) -> Tuple[_LayerBroadcaster, RowPartition]:
    """Does a NOOP broadcast of a_1.

      *-ac_0-->*
      |        |
     a_1      c_1
      |        |
      V        V
      *-ac_1-->*

  Note that by definition this cannot fail: there is always a well-defined
  NOOP broadcast. This is usually intended as half of broadcasting two shapes
  together.
  Args:
    ac_0: previous LayerBroadcaster
    a_1: previous RowPartition

  Returns:
    [ac_1, c_1] where ac_1 is the next LayerBroadcaster, and c_1 is the
    broadcast RowPartition
  """
    c_1 = ac_0.broadcast_row_partition(a_1)
    old_value_rowids = array_ops.gather(ac_0.gather_index, c_1.value_rowids())
    old_row_starts = array_ops.gather(a_1.row_splits(), old_value_rowids)
    gather_index = old_row_starts + c_1.offsets_in_rows()
    return [_LayerBroadcaster.from_gather_index(gather_index), c_1]