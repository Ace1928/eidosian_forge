import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_USER_TIMEOUT, 4)
class TCPOptionUserTimeout(TCPOption):
    _PACK_STR = '!BBH'

    def __init__(self, granularity, user_timeout, kind=None, length=None):
        super(TCPOptionUserTimeout, self).__init__(kind, length)
        self.granularity = granularity
        self.user_timeout = user_timeout

    @classmethod
    def parse(cls, buf):
        _, _, body = struct.unpack_from(cls._PACK_STR, buf)
        granularity = body >> 15
        user_timeout = body & 32767
        return (cls(granularity, user_timeout, cls.cls_kind, cls.cls_length), buf[cls.cls_length:])

    def serialize(self):
        body = self.granularity << 15 | self.user_timeout
        return struct.pack(self._PACK_STR, self.kind, self.length, body)