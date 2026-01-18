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
class _GatherLayerBroadcaster(_LayerBroadcaster):
    """Implements _LayerBroadcaster with an explicit gather_index.

  For example, suppose that the source shape is:
  [*],[*,*]
  And the target shape is:
  [*],[*,*],[*],[*,*]
  Then, this can be represented with a map:
  [0,1,2,0,1,2]

  """

    def __init__(self, gather_index):
        gather_index = ops.convert_to_tensor(gather_index)
        if gather_index.dtype != dtypes.int64 and gather_index.dtype != dtypes.int32:
            raise ValueError('gather_index must be int64 or int32')
        self._gather_index = gather_index

    @property
    def gather_index(self):
        return self._gather_index

    def with_dtype(self, dtype):
        return _GatherLayerBroadcaster(math_ops.cast(self._gather_index, dtype))

    def with_dependencies(self, checks):
        new_gather_index = control_flow_ops.with_dependencies(checks, self._gather_index)
        return _GatherLayerBroadcaster(new_gather_index)