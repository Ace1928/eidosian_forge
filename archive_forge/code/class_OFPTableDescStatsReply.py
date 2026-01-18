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
@OFPMultipartReply.register_stats_type()
@_set_stats_type(ofproto.OFPMP_TABLE_DESC, OFPTableDesc)
@_set_msg_type(ofproto.OFPT_MULTIPART_REPLY)
class OFPTableDescStatsReply(OFPMultipartReply):
    """
    Table description reply message

    The switch responds with this message to a table description request.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    body             List of ``OFPTableDesc`` instance
    ================ ======================================================

    Example::

        @set_ev_cls(ofp_event.EventOFPTableDescStatsReply, MAIN_DISPATCHER)
        def table_desc_stats_reply_handler(self, ev):
            tables = []
            for p in ev.msg.body:
                tables.append('table_id=%d config=0x%08x properties=%s' %
                             (p.table_id, p.config, repr(p.properties)))
            self.logger.debug('OFPTableDescStatsReply received: %s', tables)
    """

    def __init__(self, datapath, type_=None, **kwargs):
        super(OFPTableDescStatsReply, self).__init__(datapath, **kwargs)