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
@OFPAction.register_action_type(ofproto.OFPAT_SET_VLAN_PCP, ofproto.OFP_ACTION_VLAN_PCP_SIZE)
class OFPActionVlanPcp(OFPAction):
    """
    Set the 802.1q priority action

    This action indicates the 802.1q priority to be set.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    vlan_pcp         VLAN priority.
    ================ ======================================================
    """

    def __init__(self, vlan_pcp):
        super(OFPActionVlanPcp, self).__init__()
        self.vlan_pcp = vlan_pcp

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, vlan_pcp = struct.unpack_from(ofproto.OFP_ACTION_VLAN_PCP_PACK_STR, buf, offset)
        assert type_ == ofproto.OFPAT_SET_VLAN_PCP
        assert len_ == ofproto.OFP_ACTION_VLAN_PCP_SIZE
        return cls(vlan_pcp)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_VLAN_PCP_PACK_STR, buf, offset, self.type, self.len, self.vlan_pcp)