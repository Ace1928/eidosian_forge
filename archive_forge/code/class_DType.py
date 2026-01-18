import abc
import builtins
import dataclasses
from typing import Type, Sequence, Optional
import numpy as np
from tensorflow.core.framework import types_pb2
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.framework import _dtypes
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.types import doc_typealias
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.types import trace
from tensorflow.core.function import trace_type
from tensorflow.tools.docs import doc_controls
from tensorflow.tsl.python.lib.core import pywrap_ml_dtypes
@tf_export('dtypes.DType', 'DType')
class DType(_dtypes.DType, trace.TraceType, trace_type.Serializable, metaclass=DTypeMeta):
    """Represents the type of the elements in a `Tensor`.

  `DType`'s are used to specify the output data type for operations which
  require it, or to inspect the data type of existing `Tensor`'s.

  Examples:

  >>> tf.constant(1, dtype=tf.int64)
  <tf.Tensor: shape=(), dtype=int64, numpy=1>
  >>> tf.constant(1.0).dtype
  tf.float32

  See `tf.dtypes` for a complete list of `DType`'s defined.
  """
    __slots__ = ['_handle_data']

    def __init__(self, type_enum, handle_data=None):
        super().__init__(type_enum)
        if handle_data is not None and (not isinstance(handle_data, HandleData)):
            raise TypeError('handle_data must be of the type HandleData')
        self._handle_data = handle_data

    @property
    def _is_ref_dtype(self):
        """Returns `True` if this `DType` represents a reference type."""
        return self._type_enum > 100

    @property
    def _as_ref(self):
        """Returns a reference `DType` based on this `DType`."""
        if self._is_ref_dtype:
            return self
        else:
            return _INTERN_TABLE[self._type_enum + 100]

    @property
    def base_dtype(self):
        """Returns a non-reference `DType` based on this `DType` (for TF1).

    Programs written for TensorFlow 2.x do not need this attribute.
    It exists only for compatibility with TensorFlow 1.x, which used
    reference `DType`s in the implementation of `tf.compat.v1.Variable`.
    In TensorFlow 2.x, `tf.Variable` is implemented without reference types.
    """
        if self._is_ref_dtype:
            return _INTERN_TABLE[self._type_enum - 100]
        else:
            return self

    @property
    def real_dtype(self):
        """Returns the `DType` corresponding to this `DType`'s real part."""
        base = self.base_dtype
        if base == complex64:
            return float32
        elif base == complex128:
            return float64
        else:
            return self

    @property
    def as_numpy_dtype(self):
        """Returns a Python `type` object based on this `DType`."""
        return _TF_TO_NP[self._type_enum]

    @property
    def min(self):
        """Returns the minimum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
        if self.is_quantized or self.base_dtype in (bool, string, complex64, complex128):
            raise TypeError(f'Cannot find minimum value of {self} with {('quantized type' if self.is_quantized else 'type')} {self.base_dtype}.')
        try:
            return np.finfo(self.as_numpy_dtype).min
        except:
            try:
                return np.iinfo(self.as_numpy_dtype).min
            except:
                if self.base_dtype == bfloat16:
                    return _np_bfloat16(float.fromhex('-0x1.FEp127'))
                elif self.base_dtype == float8_e5m2:
                    return _np_float8_e5m2(float.fromhex('-0x1.Cp15'))
                elif self.base_dtype == float8_e4m3fn:
                    return _np_float8_e4m3fn(float.fromhex('-0x1.Cp8'))
                raise TypeError(f'Cannot find minimum value of {self}.')

    @property
    def max(self):
        """Returns the maximum representable value in this data type.

    Raises:
      TypeError: if this is a non-numeric, unordered, or quantized type.

    """
        if self.is_quantized or self.base_dtype in (bool, string, complex64, complex128):
            raise TypeError(f'Cannot find maximum value of {self} with {('quantized type' if self.is_quantized else 'type')} {self.base_dtype}.')
        try:
            return np.finfo(self.as_numpy_dtype).max
        except:
            try:
                return np.iinfo(self.as_numpy_dtype).max
            except:
                if self.base_dtype == bfloat16:
                    return _np_bfloat16(float.fromhex('0x1.FEp127'))
                elif self.base_dtype == float8_e5m2:
                    return _np_float8_e5m2(float.fromhex('0x1.Cp15'))
                elif self.base_dtype == float8_e4m3fn:
                    return _np_float8_e4m3fn(float.fromhex('0x1.Cp8'))
                raise TypeError(f'Cannot find maximum value of {self}.')

    @property
    def limits(self, clip_negative=True):
        """Return intensity limits, i.e.

    (min, max) tuple, of the dtype.
    Args:
      clip_negative : bool, optional If True, clip the negative range (i.e.
        return 0 for min intensity) even if the image dtype allows negative
        values. Returns
      min, max : tuple Lower and upper intensity limits.
    """
        if self.as_numpy_dtype in dtype_range:
            min, max = dtype_range[self.as_numpy_dtype]
        else:
            raise ValueError(str(self) + ' does not have defined limits.')
        if clip_negative:
            min = 0
        return (min, max)

    def is_compatible_with(self, other):
        """Returns True if the `other` DType will be converted to this DType (TF1).

    Programs written for TensorFlow 2.x do not need this function.
    Instead, they can do equality comparison on `DType` objects directly:
    `tf.as_dtype(this) == tf.as_dtype(other)`.

    This function exists only for compatibility with TensorFlow 1.x, where it
    additionally allows conversion from a reference type (used by
    `tf.compat.v1.Variable`) to its base type.

    Args:
      other: A `DType` (or object that may be converted to a `DType`).

    Returns:
      True if a Tensor of the `other` `DType` will be implicitly converted to
      this `DType`.
    """
        other = as_dtype(other)
        return self._type_enum in (other.as_datatype_enum, other.base_dtype.as_datatype_enum)

    def is_subtype_of(self, other: trace.TraceType) -> bool:
        """See tf.types.experimental.TraceType base class."""
        return self == other

    def most_specific_common_supertype(self, types: Sequence[trace.TraceType]) -> Optional['DType']:
        """See tf.types.experimental.TraceType base class."""
        return self if all((self == other for other in types)) else None

    @doc_controls.do_not_doc_inheritable
    def placeholder_value(self, placeholder_context):
        """TensorShape does not support placeholder values."""
        raise NotImplementedError

    @classmethod
    def experimental_type_proto(cls) -> Type[types_pb2.SerializedDType]:
        """Returns the type of proto associated with DType serialization."""
        return types_pb2.SerializedDType

    @classmethod
    def experimental_from_proto(cls, proto: types_pb2.SerializedDType) -> 'DType':
        """Returns a Dtype instance based on the serialized proto."""
        return DType(proto.datatype)

    def experimental_as_proto(self) -> types_pb2.SerializedDType:
        """Returns a proto representation of the Dtype instance."""
        return types_pb2.SerializedDType(datatype=self._type_enum)

    def __eq__(self, other):
        """Returns True iff this DType refers to the same type as `other`."""
        if other is None:
            return False
        if type(other) != DType:
            try:
                other = as_dtype(other)
            except TypeError:
                return False
        return self._type_enum == other._type_enum

    def __ne__(self, other):
        """Returns True iff self != other."""
        return not self.__eq__(other)
    __hash__ = _dtypes.DType.__hash__

    def __reduce__(self):
        return (as_dtype, (self.name,))