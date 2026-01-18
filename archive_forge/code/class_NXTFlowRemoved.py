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
@NiciraHeader.register_nx_subtype(ofproto.NXT_FLOW_REMOVED)
class NXTFlowRemoved(NiciraHeader):

    def __init__(self, datapath, cookie, priority, reason, duration_sec, duration_nsec, idle_timeout, match_len, packet_count, byte_count, match):
        super(NXTFlowRemoved, self).__init__(datapath, ofproto.NXT_FLOW_REMOVED)
        self.cookie = cookie
        self.priority = priority
        self.reason = reason
        self.duration_sec = duration_sec
        self.duration_nsec = duration_nsec
        self.idle_timeout = idle_timeout
        self.match_len = match_len
        self.packet_count = packet_count
        self.byte_count = byte_count
        self.match = match

    @classmethod
    def parser(cls, datapath, buf, offset):
        cookie, priority, reason, duration_sec, duration_nsec, idle_timeout, match_len, packet_count, byte_count = struct.unpack_from(ofproto.NX_FLOW_REMOVED_PACK_STR, buf, offset)
        offset += ofproto.NX_FLOW_REMOVED_SIZE - ofproto.NICIRA_HEADER_SIZE
        match = nx_match.NXMatch.parser(buf, offset, match_len)
        return cls(datapath, cookie, priority, reason, duration_sec, duration_nsec, idle_timeout, match_len, packet_count, byte_count, match)