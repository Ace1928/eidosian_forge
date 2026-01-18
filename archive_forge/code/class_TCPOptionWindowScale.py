import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_WINDOW_SCALE, 3)
class TCPOptionWindowScale(TCPOption):
    _PACK_STR = '!BBB'

    def __init__(self, shift_cnt, kind=None, length=None):
        super(TCPOptionWindowScale, self).__init__(kind, length)
        self.shift_cnt = shift_cnt

    @classmethod
    def parse(cls, buf):
        _, _, shift_cnt = struct.unpack_from(cls._PACK_STR, buf)
        return (cls(shift_cnt, cls.cls_kind, cls.cls_length), buf[cls.cls_length:])

    def serialize(self):
        return struct.pack(self._PACK_STR, self.kind, self.length, self.shift_cnt)