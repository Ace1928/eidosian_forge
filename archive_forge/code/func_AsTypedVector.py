import array
import contextlib
import enum
import struct
@property
def AsTypedVector(self):
    if self.IsTypedVector:
        return TypedVector(self._Indirect(), self._byte_width, Type.ToTypedVectorElementType(self._type))
    else:
        raise self._ConvertError('TYPED_VECTOR')