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
@_set_msg_type(ofproto.OFPT_QUEUE_GET_CONFIG_REQUEST)
class OFPQueueGetConfigRequest(MsgBase):
    """
    Queue configuration request message

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    port             Port to be queried (OFPP_ANY to all configured queues)
    ================ ======================================================

    Example::

        def send_queue_get_config_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPQueueGetConfigRequest(datapath, ofp.OFPP_ANY)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, port):
        super(OFPQueueGetConfigRequest, self).__init__(datapath)
        self.port = port

    def _serialize_body(self):
        msg_pack_into(ofproto.OFP_QUEUE_GET_CONFIG_REQUEST_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.port)