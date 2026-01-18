import array
import contextlib
import enum
import struct
@staticmethod
def PackedType(buf, parent_width, packed_type):
    byte_width, type_ = Type.Unpack(packed_type)
    return Ref(buf, parent_width, byte_width, type_)