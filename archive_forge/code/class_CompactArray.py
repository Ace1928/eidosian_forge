import struct
from struct import error
from .abstract import AbstractType
class CompactArray(Array):

    def encode(self, items):
        if items is None:
            return UnsignedVarInt32.encode(0)
        return b''.join([UnsignedVarInt32.encode(len(items) + 1)] + [self.array_of.encode(item) for item in items])

    def decode(self, data):
        length = UnsignedVarInt32.decode(data) - 1
        if length == -1:
            return None
        return [self.array_of.decode(data) for _ in range(length)]