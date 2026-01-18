import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
class TCPOptionUnknown(TCPOption):
    _PACK_STR = '!BB'

    def __init__(self, value, kind, length):
        super(TCPOptionUnknown, self).__init__(kind, length)
        self.value = value if value is not None else b''

    @classmethod
    def parse(cls, buf):
        kind, length = struct.unpack_from(cls._PACK_STR, buf)
        value = buf[2:length]
        return (cls(value, kind, length), buf[length:])

    def serialize(self):
        self.length = self.WITH_BODY_OFFSET + len(self.value)
        return struct.pack(self._PACK_STR, self.kind, self.length) + self.value