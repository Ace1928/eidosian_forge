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
@_set_stats_type(ofproto.OFPMP_AGGREGATE, OFPAggregateStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPAggregateStatsRequest(OFPFlowStatsRequestBase):
    """
    Aggregate flow statistics request message

    The controller uses this message to query aggregate flow statictics.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    table_id         ID of table to read
    out_port         Require matching entries to include this as an output
                     port
    out_group        Require matching entries to include this as an output
                     group
    cookie           Require matching entries to contain this cookie value
    cookie_mask      Mask used to restrict the cookie bits that must match
    match            Instance of ``OFPMatch``
    ================ ======================================================

    Example::

        def send_aggregate_stats_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            cookie = cookie_mask = 0
            match = ofp_parser.OFPMatch(in_port=1)
            req = ofp_parser.OFPAggregateStatsRequest(datapath, 0,
                                                      ofp.OFPTT_ALL,
                                                      ofp.OFPP_ANY,
                                                      ofp.OFPG_ANY,
                                                      cookie, cookie_mask,
                                                      match)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags, table_id, out_port, out_group, cookie, cookie_mask, match, type_=None):
        super(OFPAggregateStatsRequest, self).__init__(datapath, flags, table_id, out_port, out_group, cookie, cookie_mask, match)