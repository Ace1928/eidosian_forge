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
class _LayerBroadcaster(abc.ABC):
    """A broadcaster of a single layer.

  Although this class does not literally contain a gather_index, the reference
  implementation is defined through a gather_index. Thus, any subclasses should
  first define the gather_index property. Other functions can be overridden
  for optimization, but it should not change the behavior.
  """

    @property
    @abc.abstractmethod
    def gather_index(self):
        """Returns a 1D tensor.

    The size of the 1D tensor is equal to the destination size.

    The ith element of the result is the index of the source of the ith element.
    """
        pass

    @property
    def dtype(self):
        """Returns the dtype of the broadcast."""
        return self.gather_index.dtype

    @abc.abstractmethod
    def with_dtype(self, dtype):
        """Returns an identical _LayerBroadcaster with a different dtype."""
        pass

    def __repr__(self):
        return str(self.gather_index)

    @classmethod
    def from_gather_index(cls, gather_index):
        """Create a broadcaster from a gather_index."""
        return _GatherLayerBroadcaster(gather_index)

    @classmethod
    def first_layer(cls, nrows_source, nrows_target):
        """Create a broadcaster from a gather_index."""
        gather_index = _first_layer_gather_index(nrows_source, nrows_target)
        return _LayerBroadcaster.from_gather_index(gather_index)

    @classmethod
    def get_singleton_broadcaster(cls, target_size):
        """Broadcast from 1 element to target_size elements."""
        return _LayerBroadcaster.from_gather_index(array_ops.zeros(target_size, dtype=target_size.dtype))

    @abc.abstractmethod
    def with_dependencies(self, checks):
        """Add dependencies to a _LayerBroadcaster.

    Args:
      checks: a list of ops that need to be run before any tensors from the
        Broadcaster are used.

    Returns:
      a copy of this _LayerBroadcaster with dependencies added.
    """
        pass

    @classmethod
    def get_identity_broadcaster(cls, nvals, dtype=None):
        """Create an identity broadcaster.

    TODO(martinz): an identity broadcaster can be far more efficient than a
    generic broadcaster. Add an optimized implementation.
    Args:
      nvals: the number of values for the broadcaster.
      dtype: the dtype of the broadcaster, or None to use the dtype of nvals.

    Returns:
      an identity broadcaster from [0....nvals-1] to [0...nvals-1]
    """
        return _GatherLayerBroadcaster(math_ops.range(nvals, dtype=dtype))

    def broadcast_tensor(self, tensor):
        """Broadcast from a dense tensor.

    It is assumed that the first axis of the dense tensor is indexed by the
    source shape, and at the end, the first axis of the dense tensor is
    indexed by the destination shape.

    Args:
      tensor: a dense tensor.

    Returns:
      A dense tensor.
    """
        return array_ops.gather(tensor, self.gather_index)

    def dest_nrows(self):
        """Return the number of rows in the resulting gather, or None if tiling."""
        return math_ops.cast(array_ops.shape(self.gather_index)[0], dtype=self.dtype)

    def broadcast_row_partition(self, rp):
        """Return a new shape where the rows are broadcasted.

        *--self--->*
        |          |
        rp       result
        |          |
        V          V
        *--------->*

    This is equivalent to:
      return RowPartition.from_row_lengths(self.broadcast(rp.row_lengths()))

    However, if the shape has uniform row length, then that property is
    maintained.

    Args:
      rp: a row partition.

    Returns:
      a RowPartition representing a broadcast version of this row partition.
    """
        if not rp.is_uniform():
            return RowPartition.from_row_lengths(self.broadcast_tensor(rp.row_lengths()))
        else:
            return RowPartition.from_uniform_row_length(rp.uniform_row_length(), nvals=rp.uniform_row_length() * self.dest_nrows(), nrows=self.dest_nrows())

    def next_layer(self, original_rp, broadcast_rp):
        """Create the next layer gather_index whether or not a broadcast happens.

       *---------self------->*
       |                     |
    original_rp           broadcast_rp
       |                     |
      \\|/                   \\|/
       *--next_broadcaster-->*
    Args:
      original_rp: the original row partition.
      broadcast_rp: the target row partition.

    Returns:
      the gather_index for next_broadcaster.

    """
        gather_index = _next_layer_gather_index(self, original_rp, broadcast_rp)
        return _LayerBroadcaster.from_gather_index(gather_index)