import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_TIMESTAMPS, 10)
class TCPOptionTimestamps(TCPOption):
    _PACK_STR = '!BBII'

    def __init__(self, ts_val, ts_ecr, kind=None, length=None):
        super(TCPOptionTimestamps, self).__init__(kind, length)
        self.ts_val = ts_val
        self.ts_ecr = ts_ecr

    @classmethod
    def parse(cls, buf):
        _, _, ts_val, ts_ecr = struct.unpack_from(cls._PACK_STR, buf)
        return (cls(ts_val, ts_ecr, cls.cls_kind, cls.cls_length), buf[cls.cls_length:])

    def serialize(self):
        return struct.pack(self._PACK_STR, self.kind, self.length, self.ts_val, self.ts_ecr)