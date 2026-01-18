import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
class OFPTime(StringifyMixin):

    def __init__(self, seconds=None, nanoseconds=None):
        self.seconds = seconds
        self.nanoseconds = nanoseconds

    @classmethod
    def parser(cls, buf, offset):
        cls_ = cls()
        cls_.seconds, cls_.nanoseconds = struct.unpack_from(ofproto.OFP_TIME_PACK_STR, buf, offset)
        return cls_

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_TIME_PACK_STR, buf, offset, self.seconds, self.nanoseconds)
        return ofproto.OFP_TIME_SIZE