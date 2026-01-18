import struct
import logging
from os_ken.lib import stringify
from . import packet_base
from . import packet_utils
from . import bgp
from . import openflow
from . import zebra
@TCPOption.register(TCP_OPTION_KIND_AUTHENTICATION, 4)
class TCPOptionAuthentication(TCPOption):
    _PACK_STR = '!BBBB'

    def __init__(self, key_id, r_next_key_id, mac, kind=None, length=None):
        super(TCPOptionAuthentication, self).__init__(kind, length)
        self.key_id = key_id
        self.r_next_key_id = r_next_key_id
        self.mac = mac

    @classmethod
    def parse(cls, buf):
        _, length, key_id, r_next_key_id = struct.unpack_from(cls._PACK_STR, buf)
        mac = buf[4:length]
        return (cls(key_id, r_next_key_id, mac, cls.cls_kind, length), buf[length:])

    def serialize(self):
        self.length = self.cls_length + len(self.mac)
        return struct.pack(self._PACK_STR, self.kind, self.length, self.key_id, self.r_next_key_id) + self.mac