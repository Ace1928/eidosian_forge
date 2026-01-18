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
@OFPAction.register_action_type(ofproto.OFPAT_STRIP_VLAN, ofproto.OFP_ACTION_HEADER_SIZE)
class OFPActionStripVlan(OFPAction):
    """
    Strip the 802.1q header action

    This action indicates the 802.1q priority to be striped.
    """

    def __init__(self):
        super(OFPActionStripVlan, self).__init__()

    @classmethod
    def parser(cls, buf, offset):
        type_, len_ = struct.unpack_from(ofproto.OFP_ACTION_HEADER_PACK_STR, buf, offset)
        assert type_ == ofproto.OFPAT_STRIP_VLAN
        assert len_ == ofproto.OFP_ACTION_HEADER_SIZE
        return cls()