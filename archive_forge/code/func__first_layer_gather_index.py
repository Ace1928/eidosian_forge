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
def _first_layer_gather_index(nrows_source, nrows_target):
    """Return the first layer gather_index.

  Args:
    nrows_source: the number of rows in the source.
    nrows_target: the number of rows in the target.

  Returns:
    A tensor, usable as a gather_index for a _LayerBroadcaster.
  """

    def gi_broadcast_first():
        return array_ops.zeros(nrows_target, dtype=nrows_target.dtype)

    def gi_no_broadcast_first():
        gather_index = math_ops.range(nrows_target, dtype=nrows_target.dtype)
        return gather_index
    do_broadcast = math_ops.equal(nrows_source, constant_op.constant(1, nrows_source.dtype))
    nrows_equal = math_ops.equal(nrows_source, nrows_target)
    can_broadcast = check_ops.assert_equal(math_ops.logical_or(do_broadcast, nrows_equal), True, message='Cannot broadcast')
    gather_index = cond.cond(do_broadcast, true_fn=gi_broadcast_first, false_fn=gi_no_broadcast_first)
    return control_flow_ops.with_dependencies([can_broadcast], gather_index)