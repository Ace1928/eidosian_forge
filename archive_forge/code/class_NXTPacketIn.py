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
@NiciraHeader.register_nx_subtype(ofproto.NXT_PACKET_IN)
class NXTPacketIn(NiciraHeader):

    def __init__(self, datapath, buffer_id, total_len, reason, table_id, cookie, match_len, match, frame):
        super(NXTPacketIn, self).__init__(datapath, ofproto.NXT_PACKET_IN)
        self.buffer_id = buffer_id
        self.total_len = total_len
        self.reason = reason
        self.table_id = table_id
        self.cookie = cookie
        self.match_len = match_len
        self.match = match
        self.frame = frame

    @classmethod
    def parser(cls, datapath, buf, offset):
        buffer_id, total_len, reason, table_id, cookie, match_len = struct.unpack_from(ofproto.NX_PACKET_IN_PACK_STR, buf, offset)
        offset += ofproto.NX_PACKET_IN_SIZE - ofproto.NICIRA_HEADER_SIZE
        match = nx_match.NXMatch.parser(buf, offset, match_len)
        offset += (match_len + 7) // 8 * 8
        frame = buf[offset:]
        if total_len < len(frame):
            frame = frame[:total_len]
        return cls(datapath, buffer_id, total_len, reason, table_id, cookie, match_len, match, frame)