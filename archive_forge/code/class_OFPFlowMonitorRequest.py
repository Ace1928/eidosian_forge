import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_4 as ofproto
@_set_stats_type(ofproto.OFPMP_FLOW_MONITOR, OFPFlowUpdateHeader)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPFlowMonitorRequest(OFPFlowMonitorRequestBase):
    """
    Flow monitor request message

    The controller uses this message to query flow monitors.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    monitor_id       Controller-assigned ID for this monitor
    out_port         Require matching entries to include this as an output
                     port
    out_group        Require matching entries to include this as an output
                     group
    monitor_flags    Bitmap of the following flags.

                     | OFPFMF_INITIAL
                     | OFPFMF_ADD
                     | OFPFMF_REMOVED
                     | OFPFMF_MODIFY
                     | OFPFMF_INSTRUCTIONS
                     | OFPFMF_NO_ABBREV
                     | OFPFMF_ONLY_OWN
    table_id         ID of table to monitor
    command          One of the following values.

                     | OFPFMC_ADD
                     | OFPFMC_MODIFY
                     | OFPFMC_DELETE
    match            Instance of ``OFPMatch``
    ================ ======================================================

    Example::

        def send_flow_monitor_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            monitor_flags = [ofp.OFPFMF_INITIAL, ofp.OFPFMF_ONLY_OWN]
            match = ofp_parser.OFPMatch(in_port=1)
            req = ofp_parser.OFPFlowMonitorRequest(datapath, 0, 10000,
                                                   ofp.OFPP_ANY, ofp.OFPG_ANY,
                                                   monitor_flags,
                                                   ofp.OFPTT_ALL,
                                                   ofp.OFPFMC_ADD, match)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, monitor_id=0, out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY, monitor_flags=0, table_id=ofproto.OFPTT_ALL, command=ofproto.OFPFMC_ADD, match=None, type_=None):
        if match is None:
            match = OFPMatch()
        super(OFPFlowMonitorRequest, self).__init__(datapath, flags, monitor_id, out_port, out_group, monitor_flags, table_id, command, match)