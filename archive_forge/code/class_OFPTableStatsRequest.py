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
@_set_stats_type(ofproto.OFPMP_TABLE, OFPTableStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPTableStatsRequest(OFPMultipartRequest):
    """
    Table statistics request message

    The controller uses this message to query flow table statictics.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    ================ ======================================================

    Example::

        def send_table_stats_request(self, datapath):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPTableStatsRequest(datapath, 0)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, type_=None):
        super(OFPTableStatsRequest, self).__init__(datapath, flags)