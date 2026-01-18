import collections
import numpy as np
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import tf2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export('SparseTensorSpec')
@type_spec_registry.register('tf.SparseTensorSpec')
class SparseTensorSpec(type_spec.BatchableTypeSpec):
    """Type specification for a `tf.sparse.SparseTensor`."""
    __slots__ = ['_shape', '_dtype']
    value_type = property(lambda self: SparseTensor)

    def __init__(self, shape=None, dtype=dtypes.float32):
        """Constructs a type specification for a `tf.sparse.SparseTensor`.

    Args:
      shape: The dense shape of the `SparseTensor`, or `None` to allow any dense
        shape.
      dtype: `tf.DType` of values in the `SparseTensor`.
    """
        self._shape = tensor_shape.as_shape(shape)
        self._dtype = dtypes.as_dtype(dtype)

    def _serialize(self):
        return (self._shape, self._dtype)

    @property
    def dtype(self):
        """The `tf.dtypes.DType` specified by this type for the SparseTensor."""
        return self._dtype

    @property
    def shape(self):
        """The `tf.TensorShape` specified by this type for the SparseTensor."""
        return self._shape

    @property
    def _component_specs(self):
        rank = self._shape.ndims
        num_values = None
        return [tensor_spec.TensorSpec([num_values, rank], dtypes.int64), tensor_spec.TensorSpec([num_values], self._dtype), tensor_spec.TensorSpec([rank], dtypes.int64)]

    def _to_components(self, value):
        if isinstance(value, SparseTensorValue):
            value = SparseTensor.from_value(value)
        return [value.indices, value.values, value.dense_shape]

    def _from_components(self, tensor_list):
        if all((isinstance(t, np.ndarray) for t in tensor_list)) and (not tf2.enabled()):
            return SparseTensorValue(*tensor_list)
        else:
            result = SparseTensor(*tensor_list)
            result._dense_shape_default = result._dense_shape_default.merge_with(self._shape)
            return result

    @property
    def _flat_tensor_specs(self):
        return [tensor_spec.TensorSpec(None, dtypes.variant)]

    def _to_tensor_list(self, value):
        value = SparseTensor.from_value(value)
        return [gen_sparse_ops.serialize_sparse(value.indices, value.values, value.dense_shape, out_type=dtypes.variant)]

    def _to_batched_tensor_list(self, value):
        dense_shape = tensor_util.constant_value_as_shape(value.dense_shape)
        if self._shape.merge_with(dense_shape).ndims == 0:
            raise ValueError(f'Unbatching a sparse tensor is only supported for rank >= 1. Obtained input: {value}.')
        return [gen_sparse_ops.serialize_many_sparse(value.indices, value.values, value.dense_shape, out_type=dtypes.variant)]

    def _from_compatible_tensor_list(self, tensor_list):
        tensor_list = gen_sparse_ops.deserialize_sparse(tensor_list[0], self._dtype)
        indices, values, dense_shape = tensor_list
        rank = self._shape.ndims
        indices.set_shape([None, rank])
        if self._shape.is_fully_defined():
            dense_shape = ops.convert_to_tensor(self._shape, dtype=dtypes.int64, name='shape')
        elif self._shape.rank is not None and any((dim.value is not None for dim in self._shape.dims)):
            pieces = array_ops_stack.unstack(dense_shape, num=self._shape.rank)
            for i, dim in enumerate(self._shape.dims):
                if dim.value is not None:
                    pieces[i] = constant_op.constant(dim.value, dense_shape.dtype)
            dense_shape = array_ops_stack.stack(pieces)
        else:
            dense_shape.set_shape([rank])
        return SparseTensor(indices, values, dense_shape)

    def _batch(self, batch_size):
        return SparseTensorSpec(tensor_shape.TensorShape([batch_size]).concatenate(self._shape), self._dtype)

    def _unbatch(self):
        if self._shape.ndims == 0:
            raise ValueError('Unbatching a tensor is only supported for rank >= 1')
        return SparseTensorSpec(self._shape[1:], self._dtype)

    def _to_legacy_output_types(self):
        return self._dtype

    def _to_legacy_output_shapes(self):
        return self._shape

    def _to_legacy_output_classes(self):
        return SparseTensor

    @classmethod
    def from_value(cls, value):
        if isinstance(value, SparseTensor):
            return cls(value.shape, value.dtype)
        if isinstance(value, SparseTensorValue):
            if isinstance(value.values, np.ndarray):
                return cls(value.dense_shape, value.values.dtype)
            else:
                return cls.from_value(SparseTensor.from_value(value))
        else:
            raise TypeError(f'Expected SparseTensor or SparseTensorValue. Received: {value} of type {type(value).__name__}.')