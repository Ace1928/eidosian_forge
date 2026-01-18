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
@_set_stats_type(ofproto.OFPMP_PORT_DESC, OFPPort)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPPortDescStatsReply(OFPMultipartReply):
    """
    Port description reply message

    The switch responds with this message to a port description request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPPort`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
        def port_desc_stats_reply_handler(self, ev):
            ports = []
            for p in ev.msg.body:
                ports.append('port_no=%d hw_addr=%s name=%s config=0x%08x '
                             'state=0x%08x curr=0x%08x advertised=0x%08x '
                             'supported=0x%08x peer=0x%08x curr_speed=%d '
                             'max_speed=%d' %
                             (p.port_no, p.hw_addr,
                              p.name, p.config,
                              p.state, p.curr, p.advertised,
                              p.supported, p.peer, p.curr_speed,
                              p.max_speed))
            self.logger.debug('OFPPortDescStatsReply received: %s', ports)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPPortDescStatsReply, self).__init__(datapath, **kwargs)