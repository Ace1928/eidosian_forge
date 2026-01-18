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
class OFPFlowStats(StringifyMixin):

    def __init__(self, table_id=None, duration_sec=None, duration_nsec=None, priority=None, idle_timeout=None, hard_timeout=None, flags=None, cookie=None, packet_count=None, byte_count=None, match=None, instructions=None, length=None):
        super(OFPFlowStats, self).__init__()
        self.table_id = table_id
        self.duration_sec = duration_sec
        self.duration_nsec = duration_nsec
        self.priority = priority
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self.flags = flags
        self.cookie = cookie
        self.packet_count = packet_count
        self.byte_count = byte_count
        self.match = match
        self.instructions = instructions
        self.length = length

    @classmethod
    def parser(cls, buf, offset):
        flow_stats = cls()
        flow_stats.length, flow_stats.table_id, flow_stats.duration_sec, flow_stats.duration_nsec, flow_stats.priority, flow_stats.idle_timeout, flow_stats.hard_timeout, flow_stats.flags, flow_stats.cookie, flow_stats.packet_count, flow_stats.byte_count = struct.unpack_from(ofproto.OFP_FLOW_STATS_0_PACK_STR, buf, offset)
        offset += ofproto.OFP_FLOW_STATS_0_SIZE
        flow_stats.match = OFPMatch.parser(buf, offset)
        match_length = utils.round_up(flow_stats.match.length, 8)
        inst_length = flow_stats.length - (ofproto.OFP_FLOW_STATS_SIZE - ofproto.OFP_MATCH_SIZE + match_length)
        offset += match_length
        instructions = []
        while inst_length > 0:
            inst = OFPInstruction.parser(buf, offset)
            instructions.append(inst)
            offset += inst.len
            inst_length -= inst.len
        flow_stats.instructions = instructions
        return flow_stats