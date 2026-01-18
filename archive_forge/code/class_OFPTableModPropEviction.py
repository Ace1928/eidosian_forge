import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@OFPTableModProp.register_type(ofproto.OFPTMPT_EVICTION)
class OFPTableModPropEviction(OFPTableModProp):

    def __init__(self, type_=None, length=None, flags=None):
        self.type = type_
        self.length = length
        self.flags = flags

    @classmethod
    def parser(cls, buf):
        eviction = cls()
        eviction.type, eviction.length, eviction.flags = struct.unpack_from(ofproto.OFP_TABLE_MOD_PROP_EVICTION_PACK_STR, buf, 0)
        return eviction

    def serialize(self):
        self.length = ofproto.OFP_TABLE_MOD_PROP_EVICTION_SIZE
        buf = bytearray()
        msg_pack_into(ofproto.OFP_TABLE_MOD_PROP_EVICTION_PACK_STR, buf, 0, self.type, self.length, self.flags)
        return buf