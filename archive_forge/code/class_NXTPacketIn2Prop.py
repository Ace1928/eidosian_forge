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
class NXTPacketIn2Prop(OFPPropBase):
    _TYPES = {}
    _PROP_TYPE = None

    def __init__(self, type_=None, length=None, data=None):
        super(NXTPacketIn2Prop, self).__init__(type_, length)
        self.data = data

    @classmethod
    def parse(cls, buf):
        type_, length = struct.unpack_from(cls._PACK_STR, buf, 0)
        rest = buf[utils.round_up(length, 8):]
        try:
            subcls = cls._TYPES[type_]
        except KeyError:
            subcls = OFPPropUnknown
        prop = subcls.sub_parser(buf, type, length)
        prop.type = type_
        prop.length = length
        return (prop, rest)

    def serialize_body(self):
        return self.data

    @classmethod
    def sub_parser(cls, buf, type, length):
        p = cls(type, length - 4, buf[4:length])
        return p