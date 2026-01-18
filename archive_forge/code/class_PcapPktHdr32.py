import struct
from struct import calcsize
class PcapPktHdr32(object):
    _PACK_STR = '!II'
    _SIZE = 16

    def __init__(self, ts, caplen, len_):
        self.ts = ts
        self.caplen = caplen
        self.len = len_

    @classmethod
    def parser(cls, buf, offset):
        ts = SfTimeval32.parser(buf, offset)
        offset += SfTimeval32._SIZE
        caplen, len_ = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(ts, caplen, len_)
        return msg