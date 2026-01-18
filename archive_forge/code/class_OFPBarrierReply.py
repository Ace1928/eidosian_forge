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
@_register_parser
@_set_msg_type(ofproto.OFPT_BARRIER_REPLY)
class OFPBarrierReply(MsgBase):
    """
    Barrier reply message

    The switch responds with this message to a barrier request.

    Example::

        @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
        def barrier_reply_handler(self, ev):
            self.logger.debug('OFPBarrierReply received')
    """

    def __init__(self, datapath):
        super(OFPBarrierReply, self).__init__(datapath)