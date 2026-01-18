import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_MAXIMUM_SEGMENT_SIZE, 4)
class TCPOptionMaximumSegmentSize(TCPOption):
    _PACK_STR = '!BBH'

    def __init__(self, max_seg_size, kind=None, length=None):
        super(TCPOptionMaximumSegmentSize, self).__init__(kind, length)
        self.max_seg_size = max_seg_size

    @classmethod
    def parse(cls, buf):
        _, _, max_seg_size = struct.unpack_from(cls._PACK_STR, buf)
        return (cls(max_seg_size, cls.cls_kind, cls.cls_length), buf[cls.cls_length:])

    def serialize(self):
        return struct.pack(self._PACK_STR, self.kind, self.length, self.max_seg_size)