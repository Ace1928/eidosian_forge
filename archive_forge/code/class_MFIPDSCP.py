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
@_set_nxm_headers([ofproto_v1_0.NXM_OF_IP_TOS])
@MFField.register_field_header([ofproto_v1_0.NXM_OF_IP_TOS])
class MFIPDSCP(MFField):
    pack_str = MF_PACK_STRING_8

    def __init__(self, header, value, mask=None):
        super(MFIPDSCP, self).__init__(header, MFIPDSCP.pack_str)
        self.value = value

    @classmethod
    def make(cls, header):
        return cls(header, MFIPDSCP.pack_str)

    def put(self, buf, offset, rule):
        return self._put(buf, offset, rule.flow.nw_tos & IP_DSCP_MASK)