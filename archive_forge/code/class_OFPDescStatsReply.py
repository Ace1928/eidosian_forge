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
@OFPMultipartReply.register_stats_type(body_single_struct=True)
@_set_stats_type(ofproto.OFPMP_DESC, OFPDescStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPDescStatsReply(OFPMultipartReply):
    """
    Description statistics reply message

    The switch responds with this message to a description statistics
    request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             Instance of ``OFPDescStats``
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPDescStatsReply, MAIN_DISPATCHER)
        def desc_stats_reply_handler(self, ev):
            body = ev.msg.body

            self.logger.debug('DescStats: mfr_desc=%s hw_desc=%s sw_desc=%s '
                              'serial_num=%s dp_desc=%s',
                              body.mfr_desc, body.hw_desc, body.sw_desc,
                              body.serial_num, body.dp_desc)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPDescStatsReply, self).__init__(datapath, **kwargs)