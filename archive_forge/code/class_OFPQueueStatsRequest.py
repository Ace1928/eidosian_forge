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
@_set_stats_type(ofproto.OFPMP_QUEUE, OFPQueueStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REQUEST)
class OFPQueueStatsRequest(OFPMultipartRequest):
    """
    Queue statistics request message

    The controller uses this message to query queue statictics.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    flags            Zero or ``OFPMPF_REQ_MORE``
    port_no          Port number to read
    queue_id         ID of queue to read
    ================ ======================================================

    Example::

        def send_queue_stats_request(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.OFPQueueStatsRequest(datapath, 0, ofp.OFPP_ANY,
                                                  ofp.OFPQ_ALL)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, flags=0, port_no=ofproto.OFPP_ANY, queue_id=ofproto.OFPQ_ALL, type_=None):
        super(OFPQueueStatsRequest, self).__init__(datapath, flags)
        self.port_no = port_no
        self.queue_id = queue_id

    def _serialize_stats_body(self):
        msg_pack_into(ofproto.OFP_QUEUE_STATS_REQUEST_PACK_STR, self.buf, ofproto.OFP_MULTIPART_REQUEST_SIZE, self.port_no, self.queue_id)