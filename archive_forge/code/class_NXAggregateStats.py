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
class NXAggregateStats(ofproto_parser.namedtuple('NXAggregateStats', ('packet_count', 'byte_count', 'flow_count'))):

    @classmethod
    def parser(cls, buf, offset):
        agg = struct.unpack_from(ofproto.NX_AGGREGATE_STATS_REPLY_PACK_STR, buf, offset)
        stats = cls(*agg)
        stats.length = ofproto.NX_AGGREGATE_STATS_REPLY_SIZE
        return stats