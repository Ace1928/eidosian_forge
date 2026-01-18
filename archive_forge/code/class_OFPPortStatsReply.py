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
@_set_stats_type(ofproto.OFPMP_PORT_STATS, OFPPortStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPPortStatsReply(OFPMultipartReply):
    """
    Port statistics reply message

    The switch responds with this message to a port statistics request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPPortStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
        def port_stats_reply_handler(self, ev):
            ports = []
            for stat in ev.msg.body:
                ports.append('port_no=%d '
                             'rx_packets=%d tx_packets=%d '
                             'rx_bytes=%d tx_bytes=%d '
                             'rx_dropped=%d tx_dropped=%d '
                             'rx_errors=%d tx_errors=%d '
                             'rx_frame_err=%d rx_over_err=%d rx_crc_err=%d '
                             'collisions=%d duration_sec=%d duration_nsec=%d' %
                             (stat.port_no,
                              stat.rx_packets, stat.tx_packets,
                              stat.rx_bytes, stat.tx_bytes,
                              stat.rx_dropped, stat.tx_dropped,
                              stat.rx_errors, stat.tx_errors,
                              stat.rx_frame_err, stat.rx_over_err,
                              stat.rx_crc_err, stat.collisions,
                              stat.duration_sec, stat.duration_nsec))
            self.logger.debug('PortStats: %s', ports)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPPortStatsReply, self).__init__(datapath, **kwargs)