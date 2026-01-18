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
class OFPFlowUpdateHeader(OFPFlowUpdate):
    _EVENT = {}

    @staticmethod
    def register_flow_update_event(event, length):

        def _register_flow_update_event(cls):
            OFPFlowUpdateHeader._EVENT[event] = cls
            cls.cls_flow_update_event = event
            cls.cls_flow_update_length = length
            return cls
        return _register_flow_update_event

    def __init__(self, length=None, event=None):
        cls = self.__class__
        super(OFPFlowUpdateHeader, self).__init__(length, cls.cls_flow_update_event)
        self.length = length

    @classmethod
    def parser(cls, buf, offset):
        length, event = struct.unpack_from(ofproto.OFP_FLOW_UPDATE_HEADER_PACK_STR, buf, offset)
        cls_ = cls._EVENT[event]
        return cls_.parser(buf, offset)