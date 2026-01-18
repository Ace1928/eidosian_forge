import struct
from struct import error
from .abstract import AbstractType
class UnsignedVarInt32(AbstractType):

    @classmethod
    def decode(cls, data):
        value, i = (0, 0)
        while True:
            b, = struct.unpack('B', data.read(1))
            if not b & 128:
                break
            value |= (b & 127) << i
            i += 7
            if i > 28:
                raise ValueError('Invalid value {}'.format(value))
        value |= b << i
        return value

    @classmethod
    def encode(cls, value):
        value &= 4294967295
        ret = b''
        while value & 4294967168 != 0:
            b = value & 127 | 128
            ret += struct.pack('B', b)
            value >>= 7
        ret += struct.pack('B', value)
        return ret