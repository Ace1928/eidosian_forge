import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
class OFPPropBase(StringifyMixin):
    _PACK_STR = '!HH'

    def __init__(self, type_, length=None):
        self.type = type_
        self.length = length

    @classmethod
    def register_type(cls, type_):

        def _register_type(subcls):
            cls._TYPES[type_] = subcls
            return subcls
        return _register_type

    @classmethod
    def parse(cls, buf):
        type_, length = struct.unpack_from(cls._PACK_STR, buf, 0)
        rest = buf[utils.round_up(length, 8):]
        try:
            subcls = cls._TYPES[type_]
        except KeyError:
            subcls = OFPPropUnknown
        prop = subcls.parser(buf)
        prop.type = type_
        prop.length = length
        return (prop, rest)

    @classmethod
    def get_rest(cls, buf):
        type_, length = struct.unpack_from(cls._PACK_STR, buf, 0)
        offset = struct.calcsize(cls._PACK_STR)
        return buf[offset:length]

    def serialize(self):
        body = bytearray()
        body += self.serialize_body()
        self.length = len(body) + struct.calcsize(self._PACK_STR)
        buf = bytearray()
        msg_pack_into(self._PACK_STR, buf, 0, self.type, self.length)
        buf += body
        pad_len = utils.round_up(self.length, 8) - self.length
        msg_pack_into('%dx' % pad_len, buf, len(buf))
        return buf