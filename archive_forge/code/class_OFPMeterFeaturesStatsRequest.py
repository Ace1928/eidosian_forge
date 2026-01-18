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
@_set_stats_type(ofproto.OFPMP_METER_FEATURES, OFPMeterFeaturesStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPMeterFeaturesStatsRequest(OFPMultipartRequest):
    """
    Meter features statistics request message

    The controller uses this message to query the set of features of the
    metering subsystem.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    ================ ======================================================

    Example::

        def send_meter_features_stats_request(self, datapath):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPMeterFeaturesStatsRequest(datapath, 0)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, type_=None):
        super(OFPMeterFeaturesStatsRequest, self).__init__(datapath, flags)