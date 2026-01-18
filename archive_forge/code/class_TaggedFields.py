import struct
from struct import error
from .abstract import AbstractType
class TaggedFields(AbstractType):

    @classmethod
    def decode(cls, data):
        num_fields = UnsignedVarInt32.decode(data)
        ret = {}
        if not num_fields:
            return ret
        prev_tag = -1
        for i in range(num_fields):
            tag = UnsignedVarInt32.decode(data)
            if tag <= prev_tag:
                raise ValueError('Invalid or out-of-order tag {}'.format(tag))
            prev_tag = tag
            size = UnsignedVarInt32.decode(data)
            val = data.read(size)
            ret[tag] = val
        return ret

    @classmethod
    def encode(cls, value):
        ret = UnsignedVarInt32.encode(len(value))
        for k, v in value.items():
            assert isinstance(v, bytes), 'Value {} is not a byte array'.format(v)
            assert isinstance(k, int) and k > 0, 'Key {} is not a positive integer'.format(k)
            ret += UnsignedVarInt32.encode(k)
            ret += v
        return ret