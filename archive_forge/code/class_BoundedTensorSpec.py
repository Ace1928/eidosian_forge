from typing import Optional, Type
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
@type_spec_registry.register('tf.BoundedTensorSpec')
class BoundedTensorSpec(TensorSpec, trace_type.Serializable):
    """A `TensorSpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
  tf_minimum = tf.convert_to_tensor(spec.minimum, dtype=spec.dtype)
  tf_maximum = tf.convert_to_tensor(spec.maximum, dtype=spec.dtype)
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by tensors
  with values in the set {0, 1, 2}:
  ```python
  spec = tensor_spec.BoundedTensorSpec((3, 5), tf.int32, 0, 2)
  ```
  """
    __slots__ = ('_minimum', '_maximum')

    def __init__(self, shape, dtype, minimum, maximum, name=None):
        """Initializes a new `BoundedTensorSpec`.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      minimum: Number or sequence specifying the minimum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not provided or not
        broadcastable to `shape`.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
        super(BoundedTensorSpec, self).__init__(shape, dtype, name)
        if minimum is None:
            raise ValueError('`minimum` can not be None.')
        if maximum is None:
            raise ValueError('`maximum` can not be None.')
        try:
            minimum_shape = np.shape(minimum)
            common_shapes.broadcast_shape(tensor_shape.TensorShape(minimum_shape), self.shape)
        except ValueError as exception:
            raise ValueError(f'`minimum` {minimum} is not compatible with shape {self.shape}.') from exception
        try:
            maximum_shape = np.shape(maximum)
            common_shapes.broadcast_shape(tensor_shape.TensorShape(maximum_shape), self.shape)
        except ValueError as exception:
            raise ValueError(f'`maximum` {maximum} is not compatible with shape {self.shape}.') from exception
        self._minimum = np.array(minimum, dtype=self.dtype.as_numpy_dtype)
        self._minimum.setflags(write=False)
        self._maximum = np.array(maximum, dtype=self.dtype.as_numpy_dtype)
        self._maximum.setflags(write=False)

    @classmethod
    def experimental_type_proto(cls) -> Type[struct_pb2.BoundedTensorSpecProto]:
        """Returns the type of proto associated with BoundedTensorSpec serialization."""
        return struct_pb2.BoundedTensorSpecProto

    @classmethod
    def experimental_from_proto(cls, proto: struct_pb2.BoundedTensorSpecProto) -> 'BoundedTensorSpec':
        """Returns a BoundedTensorSpec instance based on the serialized proto."""
        return BoundedTensorSpec(shape=tensor_shape.TensorShape.experimental_from_proto(proto.shape), dtype=proto.dtype, minimum=tensor_util.MakeNdarray(proto.minimum), maximum=tensor_util.MakeNdarray(proto.maximum), name=proto.name if proto.name else None)

    def experimental_as_proto(self) -> struct_pb2.BoundedTensorSpecProto:
        """Returns a proto representation of the BoundedTensorSpec instance."""
        return struct_pb2.BoundedTensorSpecProto(shape=self.shape.experimental_as_proto(), dtype=self.dtype.experimental_as_proto().datatype, minimum=tensor_util.make_tensor_proto(self._minimum), maximum=tensor_util.make_tensor_proto(self._maximum), name=self.name)

    @classmethod
    def from_spec(cls, spec):
        """Returns a `TensorSpec` with the same shape and dtype as `spec`.

    If `spec` is a `BoundedTensorSpec`, then the new spec's bounds are set to
    `spec.minimum` and `spec.maximum`; otherwise, the bounds are set to
    `spec.dtype.min` and `spec.dtype.max`.

    >>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="x")
    >>> BoundedTensorSpec.from_spec(spec)
    BoundedTensorSpec(shape=(8, 3), dtype=tf.int32, name='x',
        minimum=array(-2147483648, dtype=int32),
        maximum=array(2147483647, dtype=int32))

    Args:
      spec: The `TypeSpec` used to create the new `BoundedTensorSpec`.
    """
        dtype = dtypes.as_dtype(spec.dtype)
        minimum = getattr(spec, 'minimum', dtype.min)
        maximum = getattr(spec, 'maximum', dtype.max)
        return BoundedTensorSpec(spec.shape, dtype, minimum, maximum, spec.name)

    @property
    def minimum(self):
        """Returns a NumPy array specifying the minimum bounds (inclusive)."""
        return self._minimum

    @property
    def maximum(self):
        """Returns a NumPy array specifying the maximum bounds (inclusive)."""
        return self._maximum

    def _cast(self, value, casting_context):
        if casting_context.allow_specs and isinstance(value, BoundedTensorSpec):
            assert value.is_subtype_of(self), f'Can not cast {value!r} to {self!r}'
            return self
        actual_spec = TensorSpec(shape=self.shape, dtype=self.dtype, name=self.name)
        return actual_spec._cast(value, casting_context)

    def __repr__(self):
        s = 'BoundedTensorSpec(shape={}, dtype={}, name={}, minimum={}, maximum={})'
        return s.format(self.shape, repr(self.dtype), repr(self.name), repr(self.minimum), repr(self.maximum))

    def __eq__(self, other):
        tensor_spec_eq = super(BoundedTensorSpec, self).__eq__(other)
        return tensor_spec_eq and np.allclose(self.minimum, other.minimum) and np.allclose(self.maximum, other.maximum)

    def __hash__(self):
        return hash((self._shape, self.dtype))

    def __reduce__(self):
        return (BoundedTensorSpec, (self._shape, self._dtype, self._minimum, self._maximum, self._name))

    def _serialize(self):
        return (self._shape, self._dtype, self._minimum, self._maximum, self._name)