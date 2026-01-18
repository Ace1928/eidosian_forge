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
@OFPQueueProp.register_property(ofproto.OFPQT_MAX_RATE, ofproto.OFP_QUEUE_PROP_MAX_RATE_SIZE)
class OFPQueuePropMaxRate(OFPQueueProp):

    def __init__(self, rate, property_=None, len_=None):
        super(OFPQueuePropMaxRate, self).__init__()
        self.rate = rate

    @classmethod
    def parser(cls, buf, offset):
        rate, = struct.unpack_from(ofproto.OFP_QUEUE_PROP_MAX_RATE_PACK_STR, buf, offset)
        return cls(rate)