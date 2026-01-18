import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
@_set_msg_reply(OFPVendorStatsReply)
@_set_stats_type(ofproto.OFPST_VENDOR, OFPVendorStats)
@_set_msg_type(ofproto.OFPT_STATS_REQUEST)
class OFPVendorStatsRequest(OFPStatsRequest):
    """
    Vendor statistics request message

    The controller uses this message to query vendor-specific information
    of a switch.
    """

    def __init__(self, datapath, flags, vendor, specific_data=None):
        super(OFPVendorStatsRequest, self).__init__(datapath, flags)
        self.vendor = vendor
        self.specific_data = specific_data

    def _serialize_vendor_stats(self):
        self.buf += self.specific_data

    def _serialize_stats_body(self):
        msg_pack_into(ofproto.OFP_VENDOR_STATS_MSG_PACK_STR, self.buf, ofproto.OFP_STATS_MSG_SIZE, self.vendor)
        self._serialize_vendor_stats()