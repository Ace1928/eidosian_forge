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
@_set_stats_type(ofproto.OFPMP_METER, OFPMeterStats)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPMeterStatsReply(OFPMultipartReply):
    """
    Meter statistics reply message

    The switch responds with this message to a meter statistics request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPMeterStats`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPMeterStatsReply, MAIN_DISPATCHER)
        def meter_stats_reply_handler(self, ev):
            meters = []
            for stat in ev.msg.body:
                meters.append('meter_id=0x%08x len=%d flow_count=%d '
                              'packet_in_count=%d byte_in_count=%d '
                              'duration_sec=%d duration_nsec=%d '
                              'band_stats=%s' %
                              (stat.meter_id, stat.len, stat.flow_count,
                               stat.packet_in_count, stat.byte_in_count,
                               stat.duration_sec, stat.duration_nsec,
                               stat.band_stats))
            self.logger.debug('MeterStats: %s', meters)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPMeterStatsReply, self).__init__(datapath, **kwargs)