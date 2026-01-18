import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
@_register_make
@_set_nxm_headers([ofproto_v1_0.NXM_OF_VLAN_TCI, ofproto_v1_0.NXM_OF_VLAN_TCI_W])
@MFField.register_field_header([ofproto_v1_0.NXM_OF_VLAN_TCI, ofproto_v1_0.NXM_OF_VLAN_TCI_W])
class MFVlan(MFField):
    pack_str = MF_PACK_STRING_BE16

    def __init__(self, header, value, mask=None):
        super(MFVlan, self).__init__(header, MFVlan.pack_str)
        self.value = value

    @classmethod
    def make(cls, header):
        return cls(header, MFVlan.pack_str)

    def put(self, buf, offset, rule):
        return self.putm(buf, offset, rule.flow.vlan_tci, rule.wc.vlan_tci_mask)