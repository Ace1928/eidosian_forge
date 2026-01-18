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
@OFPPortDescProp.register_type(ofproto.OFPPDPT_ETHERNET)
class OFPPortDescPropEthernet(OFPPortDescProp):

    def __init__(self, type_=None, length=None, curr=None, advertised=None, supported=None, peer=None, curr_speed=None, max_speed=None):
        self.type = type_
        self.length = length
        self.curr = curr
        self.advertised = advertised
        self.supported = supported
        self.peer = peer
        self.curr_speed = curr_speed
        self.max_speed = max_speed

    @classmethod
    def parser(cls, buf):
        ether = cls()
        ether.type, ether.length, ether.curr, ether.advertised, ether.supported, ether.peer, ether.curr_speed, ether.max_speed = struct.unpack_from(ofproto.OFP_PORT_DESC_PROP_ETHERNET_PACK_STR, buf, 0)
        return ether