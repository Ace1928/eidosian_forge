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
class OFPActionTpPort(OFPAction):

    def __init__(self, tp):
        super(OFPActionTpPort, self).__init__()
        self.tp = tp

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, tp = struct.unpack_from(ofproto.OFP_ACTION_TP_PORT_PACK_STR, buf, offset)
        assert type_ in (ofproto.OFPAT_SET_TP_SRC, ofproto.OFPAT_SET_TP_DST)
        assert len_ == ofproto.OFP_ACTION_TP_PORT_SIZE
        return cls(tp)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_TP_PORT_PACK_STR, buf, offset, self.type, self.len, self.tp)