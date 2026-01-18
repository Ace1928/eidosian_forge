import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class chunk_ecn_base(chunk, metaclass=abc.ABCMeta):
    _PACK_STR = '!BBHI'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, flags=0, length=0, low_tsn=0):
        super(chunk_ecn_base, self).__init__(self.chunk_type(), length)
        self.flags = flags
        self.low_tsn = low_tsn

    @classmethod
    def parser(cls, buf):
        _, flags, length, low_tsn = struct.unpack_from(cls._PACK_STR, buf)
        return cls(flags, length, low_tsn)

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length, self.low_tsn)
        return buf