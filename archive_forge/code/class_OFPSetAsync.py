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
@_set_msg_type(ofproto.OFPT_SET_ASYNC)
class OFPSetAsync(MsgBase):
    """
    Set asynchronous configuration message

    The controller sends this message to set the asynchronous messages that
    it wants to receive on a given OpneFlow channel.

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

        def send_set_async(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            packet_in_mask = 1 << ofp.OFPR_ACTION | 1 << ofp.OFPR_INVALID_TTL
            port_status_mask = (1 << ofp.OFPPR_ADD
                                | 1 << ofp.OFPPR_DELETE
                                | 1 << ofp.OFPPR_MODIFY)
            flow_removed_mask = (1 << ofp.OFPRR_IDLE_TIMEOUT
                                 | 1 << ofp.OFPRR_HARD_TIMEOUT
                                 | 1 << ofp.OFPRR_DELETE)
            req = ofp_parser.OFPSetAsync(datapath,
                                         [packet_in_mask, 0],
                                         [port_status_mask, 0],
                                         [flow_removed_mask, 0])
            datapath.send_msg(req)
    """

    def __init__(self, datapath, packet_in_mask, port_status_mask, flow_removed_mask):
        super(OFPSetAsync, self).__init__(datapath)
        self.packet_in_mask = packet_in_mask
        self.port_status_mask = port_status_mask
        self.flow_removed_mask = flow_removed_mask

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_ASYNC_CONFIG_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.packet_in_mask[0], self.packet_in_mask[1], self.port_status_mask[0], self.port_status_mask[1], self.flow_removed_mask[0], self.flow_removed_mask[1])