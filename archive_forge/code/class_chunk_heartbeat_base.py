import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class chunk_heartbeat_base(chunk, metaclass=abc.ABCMeta):
    _class_prefixes = ['param_']

    def __init__(self, flags=0, length=0, info=None):
        super(chunk_heartbeat_base, self).__init__(self.chunk_type(), length)
        self.flags = flags
        if info is not None:
            assert isinstance(info, param)
        self.info = info

    @classmethod
    def parser_base(cls, buf, recog):
        _, flags, length = struct.unpack_from(cls._PACK_STR, buf)
        ptype, = struct.unpack_from('!H', buf, cls._MIN_LEN)
        cls_ = recog.get(ptype)
        info = cls_.parser(buf[cls._MIN_LEN:])
        msg = cls(flags, length, info)
        return msg

    def serialize(self):
        buf = bytearray(struct.pack(self._PACK_STR, self.chunk_type(), self.flags, self.length))
        if self.info is not None:
            buf.extend(self.info.serialize())
        if 0 == self.length:
            self.length = len(buf)
            struct.pack_into('!H', buf, 2, self.length)
        return bytes(buf)