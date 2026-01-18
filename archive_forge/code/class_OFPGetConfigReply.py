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
@_set_msg_type(ofproto.OFPT_GET_CONFIG_REPLY)
class OFPGetConfigReply(MsgBase):
    """
    Get config reply message

    The switch responds to a configuration request with a get config reply
    message.

    ============= =========================================================
    Attribute     Description
    ============= =========================================================
    flags         Bitmap of the following flags.

                  | OFPC_FRAG_NORMAL
                  | OFPC_FRAG_DROP
                  | OFPC_FRAG_REASM
                  | OFPC_FRAG_MASK
    miss_send_len Max bytes of new flow that datapath should send to the
                  controller
    ============= =========================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPGetConfigReply, MAIN_DISPATCHER)
        def get_config_reply_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto
            flags = []

            if msg.flags & ofp.OFPC_FRAG_NORMAL:
                flags.append('NORMAL')
            if msg.flags & ofp.OFPC_FRAG_DROP:
                flags.append('DROP')
            if msg.flags & ofp.OFPC_FRAG_REASM:
                flags.append('REASM')
            self.logger.debug('OFPGetConfigReply received: '
                              'flags=%s miss_send_len=%d',
                              ','.join(flags), msg.miss_send_len)
    """

    def __init__(self, datapath, flags=None, miss_send_len=None):
        super(OFPGetConfigReply, self).__init__(datapath)
        self.flags = flags
        self.miss_send_len = miss_send_len

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPGetConfigReply, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.flags, msg.miss_send_len = struct.unpack_from(ofproto.OFP_SWITCH_CONFIG_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        return msg