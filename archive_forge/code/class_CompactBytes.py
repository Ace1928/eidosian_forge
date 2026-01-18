import struct
from struct import error
from .abstract import AbstractType
class CompactBytes(AbstractType):

    @classmethod
    def decode(cls, data):
        length = UnsignedVarInt32.decode(data) - 1
        if length < 0:
            return None
        value = data.read(length)
        if len(value) != length:
            raise ValueError('Buffer underrun decoding Bytes')
        return value

    @classmethod
    def encode(cls, value):
        if value is None:
            return UnsignedVarInt32.encode(0)
        else:
            return UnsignedVarInt32.encode(len(value) + 1) + value