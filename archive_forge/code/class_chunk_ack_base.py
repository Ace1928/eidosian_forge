import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class chunk_ack_base(chunk, metaclass=abc.ABCMeta):

    def __init__(self, flags=0, length=0):
        super(chunk_ack_base, self).__init__(self.chunk_type(), length)
        self.flags = flags

    @classmethod
    def parser(cls, buf):
        _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
        return cls(flags, length)

    def serialize(self):
        if 0 == self.length:
            self.length = self._MIN_LEN
        buf = struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length)
        return buf