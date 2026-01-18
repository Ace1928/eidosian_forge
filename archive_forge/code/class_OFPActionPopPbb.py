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
@OFPAction.register_action_type(ofproto.OFPAT_POP_PBB, ofproto.OFP_ACTION_HEADER_SIZE)
class OFPActionPopPbb(OFPAction):
    """
    Pop PBB action

    This action pops the outermost PBB service instance header from
    the packet.
    """

    def __init__(self, type_=None, len_=None):
        super(OFPActionPopPbb, self).__init__()

    @classmethod
    def parser(cls, buf, offset):
        type_, len_ = struct.unpack_from(ofproto.OFP_ACTION_HEADER_PACK_STR, buf, offset)
        return cls()