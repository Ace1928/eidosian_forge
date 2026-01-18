import struct
import base64
import netaddr
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib import mac
from os_ken.lib.packet import packet
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0 as ofproto
from os_ken.ofproto import nx_actions
from os_ken import utils
import logging
class NXFlowStats(StringifyMixin):

    def __init__(self):
        super(NXFlowStats, self).__init__()
        self.length = None
        self.table_id = None
        self.duration_sec = None
        self.duration_nsec = None
        self.priority = None
        self.idle_timeout = None
        self.hard_timeout = None
        self.match_len = None
        self.idle_age = None
        self.hard_age = None
        self.cookie = None
        self.packet_count = None
        self.byte_count = None

    @classmethod
    def parser(cls, buf, offset):
        original_offset = offset
        nxflow_stats = cls()
        nxflow_stats.length, nxflow_stats.table_id, nxflow_stats.duration_sec, nxflow_stats.duration_nsec, nxflow_stats.priority, nxflow_stats.idle_timeout, nxflow_stats.hard_timeout, nxflow_stats.match_len, nxflow_stats.idle_age, nxflow_stats.hard_age, nxflow_stats.cookie, nxflow_stats.packet_count, nxflow_stats.byte_count = struct.unpack_from(ofproto.NX_FLOW_STATS_PACK_STR, buf, offset)
        offset += ofproto.NX_FLOW_STATS_SIZE
        fields = []
        match_len = nxflow_stats.match_len
        match_len -= 4
        while match_len > 0:
            field = nx_match.MFField.parser(buf, offset)
            offset += field.length
            match_len -= field.length
            fields.append(field)
        nxflow_stats.fields = fields
        actions = []
        total_len = original_offset + nxflow_stats.length
        match_len = nxflow_stats.match_len
        offset += utils.round_up(match_len, 8) - match_len
        while offset < total_len:
            action = OFPAction.parser(buf, offset)
            actions.append(action)
            offset += action.len
        nxflow_stats.actions = actions
        return nxflow_stats