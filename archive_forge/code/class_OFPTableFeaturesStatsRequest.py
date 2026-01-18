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
@_set_stats_type(ofproto.OFPMP_TABLE_FEATURES, OFPTableFeaturesStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPTableFeaturesStatsRequest(OFPMultipartRequest):
    """
    Table features statistics request message

    The controller uses this message to query table features.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPTableFeaturesStats`` instances.
                     The default is [].
    ================ ======================================================
    """

    def __init__(self, datapath, flags=0, body=None, type_=None):
        body = body if body else []
        super(OFPTableFeaturesStatsRequest, self).__init__(datapath, flags)
        self.body = body

    def _serialize_stats_body(self):
        bin_body = bytearray()
        for p in self.body:
            bin_body += p.serialize()
        self.buf += bin_body