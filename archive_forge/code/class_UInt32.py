import struct
from struct import error
from .abstract import AbstractType
class UInt32(AbstractType):
    _pack = struct.Struct('>I').pack
    _unpack = struct.Struct('>I').unpack

    @classmethod
    def encode(cls, value):
        return _pack(cls._pack, value)

    @classmethod
    def decode(cls, data):
        return _unpack(cls._unpack, data.read(4))