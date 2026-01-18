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
@OFPAction.register_action_type(ofproto.OFPAT_SET_VLAN_VID, ofproto.OFP_ACTION_VLAN_VID_SIZE)
class OFPActionVlanVid(OFPAction):
    """
    Set the 802.1q VLAN id action

    This action indicates the 802.1q VLAN id to be set.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    vlan_vid         VLAN id.
    ================ ======================================================
    """

    def __init__(self, vlan_vid):
        super(OFPActionVlanVid, self).__init__()
        self.vlan_vid = vlan_vid

    @classmethod
    def parser(cls, buf, offset):
        type_, len_, vlan_vid = struct.unpack_from(ofproto.OFP_ACTION_VLAN_VID_PACK_STR, buf, offset)
        assert type_ == ofproto.OFPAT_SET_VLAN_VID
        assert len_ == ofproto.OFP_ACTION_VLAN_VID_SIZE
        return cls(vlan_vid)

    def serialize(self, buf, offset):
        msg_pack_into(ofproto.OFP_ACTION_VLAN_VID_PACK_STR, buf, offset, self.type, self.len, self.vlan_vid)