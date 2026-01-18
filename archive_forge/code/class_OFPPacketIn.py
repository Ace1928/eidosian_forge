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
@_set_msg_type(ofproto.OFPT_PACKET_IN)
class OFPPacketIn(MsgBase):
    """
    Packet-In message

    The switch sends the packet that received to the controller by this
    message.

    ============= =========================================================
    Attribute     Description
    ============= =========================================================
    buffer_id     ID assigned by datapath
    total_len     Full length of frame
    reason        Reason packet is being sent.

                  | OFPR_NO_MATCH
                  | OFPR_ACTION
                  | OFPR_INVALID_TTL
    table_id      ID of the table that was looked up
    cookie        Cookie of the flow entry that was looked up
    match         Instance of ``OFPMatch``
    data          Ethernet frame
    ============= =========================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
        def packet_in_handler(self, ev):
            msg = ev.msg
            dp = msg.datapath
            ofp = dp.ofproto

            if msg.reason == ofp.OFPR_NO_MATCH:
                reason = 'NO MATCH'
            elif msg.reason == ofp.OFPR_ACTION:
                reason = 'ACTION'
            elif msg.reason == ofp.OFPR_INVALID_TTL:
                reason = 'INVALID TTL'
            else:
                reason = 'unknown'

            self.logger.debug('OFPPacketIn received: '
                              'buffer_id=%x total_len=%d reason=%s '
                              'table_id=%d cookie=%d match=%s data=%s',
                              msg.buffer_id, msg.total_len, reason,
                              msg.table_id, msg.cookie, msg.match,
                              utils.hex_array(msg.data))
    """

    def __init__(self, datapath, buffer_id=None, total_len=None, reason=None, table_id=None, cookie=None, match=None, data=None):
        super(OFPPacketIn, self).__init__(datapath)
        self.buffer_id = buffer_id
        self.total_len = total_len
        self.reason = reason
        self.table_id = table_id
        self.cookie = cookie
        self.match = match
        self.data = data

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPPacketIn, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        msg.buffer_id, msg.total_len, msg.reason, msg.table_id, msg.cookie = struct.unpack_from(ofproto.OFP_PACKET_IN_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        msg.match = OFPMatch.parser(msg.buf, ofproto.OFP_PACKET_IN_SIZE - ofproto.OFP_MATCH_SIZE)
        match_len = utils.round_up(msg.match.length, 8)
        msg.data = msg.buf[ofproto.OFP_PACKET_IN_SIZE - ofproto.OFP_MATCH_SIZE + match_len + 2:]
        if msg.total_len < len(msg.data):
            msg.data = msg.data[:msg.total_len]
        return msg