import array
import contextlib
import enum
import struct
@property
def AsFixedTypedVector(self):
    if self.IsFixedTypedVector:
        element_type, size = Type.ToFixedTypedVectorElementType(self._type)
        return TypedVector(self._Indirect(), self._byte_width, element_type, size)
    else:
        raise self._ConvertError('FIXED_TYPED_VECTOR')