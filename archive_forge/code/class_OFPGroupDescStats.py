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
class OFPGroupDescStats(StringifyMixin):

    def __init__(self, type_=None, group_id=None, buckets=None, length=None):
        super(OFPGroupDescStats, self).__init__()
        self.type = type_
        self.group_id = group_id
        self.buckets = buckets

    @classmethod
    def parser(cls, buf, offset):
        stats = cls()
        stats.length, stats.type, stats.group_id = struct.unpack_from(ofproto.OFP_GROUP_DESC_STATS_PACK_STR, buf, offset)
        offset += ofproto.OFP_GROUP_DESC_STATS_SIZE
        stats.buckets = []
        length = ofproto.OFP_GROUP_DESC_STATS_SIZE
        while length < stats.length:
            bucket = OFPBucket.parser(buf, offset)
            stats.buckets.append(bucket)
            offset += bucket.len
            length += bucket.len
        return stats