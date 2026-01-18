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
def _register_nx_stats_type(cls):
    assert cls.cls_stats_type is not None
    assert cls.cls_stats_type not in NXStatsReply._NX_STATS_TYPES
    assert cls.cls_stats_body_cls is not None
    cls.cls_body_single_struct = body_single_struct
    NXStatsReply._NX_STATS_TYPES[cls.cls_stats_type] = cls
    return cls