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
@_set_stats_type(ofproto.OFPMP_PORT_STATS, OFPPortStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPPortStatsRequest(OFPMultipartRequest):
    """
    Port statistics request message

    The controller uses this message to query information about ports
    statistics.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    port_no          Port number to read (OFPP_ANY to all ports)
    ================ ======================================================

    Example::

        def send_port_stats_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPPortStatsRequest(datapath, 0, ofp.OFPP_ANY)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, port_no=ofproto.OFPP_ANY, type_=None):
        super(OFPPortStatsRequest, self).__init__(datapath, flags)
        self.port_no = port_no

    def _serialize_stats_body(self):
        msg_pack_into(ofproto.OFP_PORT_STATS_REQUEST_PACK_STR, self.buf, ofproto.OFP_MULTIPART_REQUEST_SIZE, self.port_no)