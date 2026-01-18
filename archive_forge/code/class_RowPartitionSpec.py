import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import segment_id_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util.tf_export import tf_export
@type_spec_registry.register('tf.RowPartitionSpec')
class RowPartitionSpec(type_spec.TypeSpec):
    """Type specification for a `tf.RowPartition`."""
    __slots__ = ['_nrows', '_nvals', '_uniform_row_length', '_dtype']
    value_type = property(lambda self: RowPartition)

    def __init__(self, nrows=None, nvals=None, uniform_row_length=None, dtype=dtypes.int64):
        """Constructs a new RowPartitionSpec.

    Args:
      nrows: The number of rows in the RowPartition, or `None` if unspecified.
      nvals: The number of values partitioned by the RowPartition, or `None` if
        unspecified.
      uniform_row_length: The number of values in each row for this
        RowPartition, or `None` if rows are ragged or row length is unspecified.
      dtype: The data type used to encode the partition.  One of `tf.int64` or
        `tf.int32`.
    """
        nrows = tensor_shape.TensorShape([nrows])
        nvals = tensor_shape.TensorShape([nvals])
        if not isinstance(uniform_row_length, tensor_shape.TensorShape):
            uniform_row_length = tensor_shape.TensorShape([uniform_row_length])
        else:
            uniform_row_length = uniform_row_length.with_rank(1)
        self._nrows = nrows
        self._nvals = nvals
        self._uniform_row_length = uniform_row_length
        self._dtype = dtypes.as_dtype(dtype)
        if self._dtype not in (dtypes.int32, dtypes.int64):
            raise ValueError('dtype must be tf.int32 or tf.int64')
        nrows = tensor_shape.dimension_value(nrows[0])
        nvals = tensor_shape.dimension_value(nvals[0])
        ncols = tensor_shape.dimension_value(uniform_row_length[0])
        if nrows == 0:
            if nvals is None:
                self._nvals = tensor_shape.TensorShape([0])
            elif nvals != 0:
                raise ValueError('nvals=%s is not compatible with nrows=%s' % (nvals, nrows))
        if ncols == 0:
            if nvals is None:
                self._nvals = tensor_shape.TensorShape([0])
            elif nvals != 0:
                raise ValueError('nvals=%s is not compatible with uniform_row_length=%s' % (nvals, uniform_row_length))
        if ncols is not None and nvals is not None:
            if ncols != 0 and nvals % ncols != 0:
                raise ValueError("nvals=%s is not compatible with uniform_row_length=%s (doesn't divide evenly)" % (nvals, ncols))
            if nrows is not None and nvals != ncols * nrows:
                raise ValueError('nvals=%s is not compatible with nrows=%s and uniform_row_length=%s' % (nvals, nrows, ncols))
            if nrows is None and ncols != 0:
                self._nrows = tensor_shape.TensorShape([nvals // ncols])
        if ncols is not None and nrows is not None and (nvals is None):
            self._nvals = tensor_shape.TensorShape([ncols * nrows])

    def is_compatible_with(self, other):
        if not super(RowPartitionSpec, self).is_compatible_with(other):
            return False
        nrows = self._nrows.merge_with(other.nrows)
        nvals = self._nvals.merge_with(other.nvals)
        ncols = self._uniform_row_length.merge_with(other.uniform_row_length)
        return self._dimensions_compatible(nrows, nvals, ncols)

    def _serialize(self):
        return (self._nrows, self._nvals, self._uniform_row_length, self._dtype)

    @classmethod
    def _deserialize(cls, serialization):
        nrows, nvals, uniform_row_length, dtype = serialization
        nrows = tensor_shape.dimension_value(nrows[0])
        nvals = tensor_shape.dimension_value(nvals[0])
        return cls(nrows, nvals, uniform_row_length, dtype)

    @property
    def nrows(self):
        return tensor_shape.dimension_value(self._nrows[0])

    @property
    def nvals(self):
        return tensor_shape.dimension_value(self._nvals[0])

    @property
    def uniform_row_length(self):
        return tensor_shape.dimension_value(self._uniform_row_length[0])

    @property
    def dtype(self):
        return self._dtype

    @property
    def _component_specs(self):
        row_splits_shape = tensor_shape.TensorShape([tensor_shape.dimension_at_index(self._nrows, 0) + 1])
        return tensor_lib.TensorSpec(row_splits_shape, self._dtype)

    def _to_components(self, value):
        return value.row_splits()

    def _from_components(self, tensor):
        return RowPartition.from_row_splits(tensor, validate=False)

    @classmethod
    def from_value(cls, value):
        if not isinstance(value, RowPartition):
            raise TypeError('Expected `value` to be a `RowPartition`')
        return cls(value.static_nrows, value.static_nvals, value.static_uniform_row_length, value.dtype)

    def __repr__(self):
        return 'RowPartitionSpec(nrows=%s, nvals=%s, uniform_row_length=%s, dtype=%r)' % (self.nrows, self.nvals, self.uniform_row_length, self.dtype)

    @staticmethod
    def _dimensions_compatible(nrows, nvals, uniform_row_length):
        """Returns true if the given dimensions are compatible."""
        nrows = tensor_shape.dimension_value(nrows[0])
        nvals = tensor_shape.dimension_value(nvals[0])
        ncols = tensor_shape.dimension_value(uniform_row_length[0])
        if nrows == 0 and nvals not in (0, None):
            return False
        if ncols == 0 and nvals not in (0, None):
            return False
        if ncols is not None and nvals is not None:
            if ncols != 0 and nvals % ncols != 0:
                return False
            if nrows is not None and nvals != ncols * nrows:
                return False
        return True

    def _merge_with(self, other):
        """Merge two RowPartitionSpecs."""
        nrows = self._nrows.merge_with(other.nrows)
        nvals = self._nvals.merge_with(other.nvals)
        ncols = self._uniform_row_length.merge_with(other.uniform_row_length)
        if not RowPartitionSpec._dimensions_compatible(nrows, nvals, ncols):
            raise ValueError('Merging incompatible RowPartitionSpecs')
        if self.dtype != other.dtype:
            raise ValueError('Merging RowPartitionSpecs with incompatible dtypes')
        return RowPartitionSpec(nrows=nrows[0], nvals=nvals[0], uniform_row_length=ncols[0], dtype=self.dtype)

    def with_dtype(self, dtype):
        nrows = tensor_shape.dimension_value(self._nrows[0])
        nvals = tensor_shape.dimension_value(self._nvals[0])
        return RowPartitionSpec(nrows, nvals, self._uniform_row_length, dtype)

    def __deepcopy__(self, memo):
        del memo
        dtype = self.dtype
        nrows = tensor_shape.dimension_value(self._nrows[0])
        nvals = tensor_shape.dimension_value(self._nvals[0])
        uniform_row_length = None if self._uniform_row_length is None else tensor_shape.dimension_value(self._uniform_row_length[0])
        return RowPartitionSpec(nrows, nvals, uniform_row_length, dtype)