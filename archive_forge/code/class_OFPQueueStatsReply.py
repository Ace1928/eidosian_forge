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
@OFPMultipartReply.register_stats_type()
@_set_stats_type(ofproto.OFPMP_QUEUE, OFPQueueStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPQueueStatsReply(OFPMultipartReply):
    """
    Queue statistics reply message

    The switch responds with this message to an aggregate flow statistics
    request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPQueueStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPQueueStatsReply, MAIN_DISPATCHER)
        def queue_stats_reply_handler(self, ev):
            queues = []
            for stat in ev.msg.body:
                queues.append('port_no=%d queue_id=%d '
                              'tx_bytes=%d tx_packets=%d tx_errors=%d '
                              'duration_sec=%d duration_nsec=%d' %
                              (stat.port_no, stat.queue_id,
                               stat.tx_bytes, stat.tx_packets, stat.tx_errors,
                               stat.duration_sec, stat.duration_nsec))
            self.logger.debug('QueueStats: %s', queues)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPQueueStatsReply, self).__init__(datapath, **kwargs)