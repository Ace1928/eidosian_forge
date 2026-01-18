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
@_set_msg_type(ofproto.OFPT_GET_ASYNC_REPLY)
class OFPGetAsyncReply(MsgBase):
    """
    Get asynchronous configuration reply message

    The switch responds with this message to a get asynchronous configuration
    request.

    ================== ====================================================
    Attribute          Description
    ================== ====================================================
    packet_in_mask     2-element array: element 0, when the controller has a
                       OFPCR_ROLE_EQUAL or OFPCR_ROLE_MASTER role. element 1,
                       OFPCR_ROLE_SLAVE role controller.
                       Bitmasks of following values.

                       | OFPR_NO_MATCH
                       | OFPR_ACTION
                       | OFPR_INVALID_TTL
    port_status_mask   2-element array.
                       Bitmasks of following values.

                       | OFPPR_ADD
                       | OFPPR_DELETE
                       | OFPPR_MODIFY
    flow_removed_mask  2-element array.
                       Bitmasks of following values.

                       | OFPRR_IDLE_TIMEOUT
                       | OFPRR_HARD_TIMEOUT
                       | OFPRR_DELETE
                       | OFPRR_GROUP_DELETE
    ================== ====================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPGetAsyncReply, MAIN_DISPATCHER)
        def get_async_reply_handler(self, ev):
            msg = ev.msg

            self.logger.debug('OFPGetAsyncReply received: '
                              'packet_in_mask=0x%08x:0x%08x '
                              'port_status_mask=0x%08x:0x%08x '
                              'flow_removed_mask=0x%08x:0x%08x',
                              msg.packet_in_mask[0],
                              msg.packet_in_mask[1],
                              msg.port_status_mask[0],
                              msg.port_status_mask[1],
                              msg.flow_removed_mask[0],
                              msg.flow_removed_mask[1])
    """

    def __init__(self, datapath, packet_in_mask=None, port_status_mask=None, flow_removed_mask=None):
        super(OFPGetAsyncReply, self).__init__(datapath)
        self.packet_in_mask = packet_in_mask
        self.port_status_mask = port_status_mask
        self.flow_removed_mask = flow_removed_mask

    @classmethod
    def parser(cls, datapath, version, msg_type, msg_len, xid, buf):
        msg = super(OFPGetAsyncReply, cls).parser(datapath, version, msg_type, msg_len, xid, buf)
        packet_in_mask_m, packet_in_mask_s, port_status_mask_m, port_status_mask_s, flow_removed_mask_m, flow_removed_mask_s = struct.unpack_from(ofproto.OFP_ASYNC_CONFIG_PACK_STR, msg.buf, ofproto.OFP_HEADER_SIZE)
        msg.packet_in_mask = [packet_in_mask_m, packet_in_mask_s]
        msg.port_status_mask = [port_status_mask_m, port_status_mask_s]
        msg.flow_removed_mask = [flow_removed_mask_m, flow_removed_mask_s]
        return msg