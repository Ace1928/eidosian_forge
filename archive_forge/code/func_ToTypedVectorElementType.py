import array
import contextlib
import enum
import struct
@staticmethod
def ToTypedVectorElementType(type_):
    if not Type.IsTypedVector(type_):
        raise ValueError('must be typed vector type')
    return Type(type_ - Type.VECTOR_INT + Type.INT)