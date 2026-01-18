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
class OFPPortModPropOptical(OFPPortModProp):

    def __init__(self, type_=None, length=None, configure=None, freq_lmda=None, fl_offset=None, grid_span=None, tx_pwr=None):
        self.type = type_
        self.length = length
        self.configure = configure
        self.freq_lmda = freq_lmda
        self.fl_offset = fl_offset
        self.grid_span = grid_span
        self.tx_pwr = tx_pwr

    def serialize(self):
        self.length = struct.calcsize(ofproto.OFP_PORT_MOD_PROP_OPTICAL_PACK_STR)
        buf = bytearray()
        msg_pack_into(ofproto.OFP_PORT_MOD_PROP_OPTICAL_PACK_STR, buf, 0, self.type, self.length, self.configure, self.freq_lmda, self.fl_offset, self.grid_span, self.tx_pwr)
        return buf