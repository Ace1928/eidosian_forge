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
@_set_nxm_headers([ofproto_v1_0.NXM_NX_TUN_ID, ofproto_v1_0.NXM_NX_TUN_ID_W])
@MFField.register_field_header([ofproto_v1_0.NXM_NX_TUN_ID, ofproto_v1_0.NXM_NX_TUN_ID_W])
class MFTunId(MFField):
    pack_str = MF_PACK_STRING_BE64

    def __init__(self, header, value, mask=None):
        super(MFTunId, self).__init__(header, MFTunId.pack_str)
        self.value = value

    @classmethod
    def make(cls, header):
        return cls(header, MFTunId.pack_str)

    def put(self, buf, offset, rule):
        return self.putm(buf, offset, rule.flow.tun_id, rule.wc.tun_id_mask)