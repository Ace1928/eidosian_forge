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
class OFPGroupFeaturesStats(ofproto_parser.namedtuple('OFPGroupFeaturesStats', ('types', 'capabilities', 'max_groups', 'actions'))):

    @classmethod
    def parser(cls, buf, offset):
        group_features = struct.unpack_from(ofproto.OFP_GROUP_FEATURES_PACK_STR, buf, offset)
        types = group_features[0]
        capabilities = group_features[1]
        max_groups = list(group_features[2:6])
        actions = list(group_features[6:10])
        stats = cls(types, capabilities, max_groups, actions)
        stats.length = ofproto.OFP_GROUP_FEATURES_SIZE
        return stats