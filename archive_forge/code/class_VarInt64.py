import struct
from struct import error
from .abstract import AbstractType
class VarInt64(AbstractType):

    @classmethod
    def decode(cls, data):
        value, i = (0, 0)
        while True:
            b = data.read(1)
            if not b & 128:
                break
            value |= (b & 127) << i
            i += 7
            if i > 63:
                raise ValueError('Invalid value {}'.format(value))
        value |= b << i
        return value >> 1 ^ -(value & 1)

    @classmethod
    def encode(cls, value):
        value &= 18446744073709551615
        v = value << 1 ^ value >> 63
        ret = b''
        while v & 18446744073709551488 != 0:
            b = value & 127 | 128
            ret += struct.pack('B', b)
            v >>= 7
        ret += struct.pack('B', v)
        return ret