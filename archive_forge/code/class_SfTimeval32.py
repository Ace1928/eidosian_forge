import struct
from struct import calcsize
class SfTimeval32(object):
    _PACK_STR = '!II'
    _SIZE = 8

    def __init__(self, tv_sec, tv_usec):
        self.tv_sec = tv_sec
        self.tv_usec = tv_usec

    @classmethod
    def parser(cls, buf, offset):
        tv_sec, tv_usec = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(tv_sec, tv_usec)
        return msg