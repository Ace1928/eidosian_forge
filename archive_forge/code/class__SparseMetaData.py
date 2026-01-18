from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.summary import summary
from tensorflow.python.training import queue_runner
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _SparseMetaData:
    """Store information about the Tensor: Is it sparse?, map_op, and rank."""

    def __init__(self, sparse, map_op, rank):
        """Create the metadata.

    Args:
      sparse: Python boolean.
      map_op: The `Operation` that created the `SparseTensorsMap` in question.
        This Op contains information about the underlying Map object and the
        dtype of the original data.
      rank: The statically known rank of the `SparseTensor`.
    """
        self._sparse = sparse
        self._map_op = map_op
        self._rank = tensor_shape.as_dimension(rank)

    def __eq__(self, other):
        if self.sparse != other.sparse:
            return False
        if not self.sparse:
            return True
        if (self.map_op is not None) != (other.map_op is not None):
            return False
        if self.map_op != other.map_op:
            return False
        if not self.rank.is_compatible_with(other.rank):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return '[SparseMetaData(%s, %s, %s)]' % (self.sparse, self.map_op.name, self.rank)

    def merge_with(self, other):
        if self != other:
            raise ValueError('SparseMetaData objects are incompatible: %s vs. %s' % (self, other))
        if self.sparse:
            self.rank.merge_with(other.rank)
        return self

    @property
    def map_op(self):
        return self._map_op

    @property
    def sparse(self):
        return self._sparse

    @property
    def rank(self):
        return self._rank