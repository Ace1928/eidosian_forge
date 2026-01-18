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
@OFPAction.register_action_type(ofproto.OFPAT_SET_TP_DST, ofproto.OFP_ACTION_TP_PORT_SIZE)
class OFPActionSetTpDst(OFPActionTpPort):
    """
    Set the TCP/UDP destination port action

    This action indicates the TCP/UDP destination port to be set.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    tp               TCP/UDP port.
    ================ ======================================================
    """

    def __init__(self, tp):
        super(OFPActionSetTpDst, self).__init__(tp)