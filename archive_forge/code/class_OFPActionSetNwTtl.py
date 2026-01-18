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
@OFPAction.register_action_type(ofproto.OFPAT_SET_NW_TTL, ofproto.OFP_ACTION_NW_TTL_SIZE)
class OFPActionSetNwTtl(OFPAction):
    """
    Set IP TTL action

    This action sets the IP TTL.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    nw_ttl           IP TTL
    ================ ======================================================
    """

    def __init__(self, nw_ttl, type_=None, len_=None):
        super(OFPActionSetNwTtl, self).__init__()
        self.nw_ttl = nw_ttl

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, nw_ttl = struct.unpack_from(ofproto.OFP_ACTION_NW_TTL_PACK_STR, buf, offset)
        return cls(nw_ttl)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_NW_TTL_PACK_STR, buf, offset, self.type, self.len, self.nw_ttl)