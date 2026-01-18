import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
@_set_stats_type(ofproto.OFPMP_CONTROLLER_STATUS, OFPControllerStatusStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPControllerStatusStatsRequest(OFPMultipartRequest):
    """
    Controller status multipart request message

    The controller uses this message to request the status, the roles
    and the control channels of other controllers configured on the switch.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    ================ ======================================================

    Example::

        def send_controller_status_multipart_request(self, datapath):
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPPortDescStatsRequest(datapath, 0)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, type_=None):
        super(OFPControllerStatusStatsRequest, self).__init__(datapath, flags)