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
@OFPAction.register_action_type(ofproto.OFPAT_SET_NW_DST, ofproto.OFP_ACTION_NW_ADDR_SIZE)
class OFPActionSetNwDst(OFPActionNwAddr):
    """
    Set the IP destination address action

    This action indicates the IP destination address to be set.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    nw_addr          IP address.
    ================ ======================================================
    """

    def __init__(self, nw_addr):
        super(OFPActionSetNwDst, self).__init__(nw_addr)