import array
import contextlib
import enum
import struct
@staticmethod
def IsTypedVectorElementType(type_):
    return Type.INT <= type_ <= Type.STRING or type_ == Type.BOOL