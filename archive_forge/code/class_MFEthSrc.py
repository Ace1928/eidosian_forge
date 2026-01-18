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
@_set_nxm_headers([ofproto_v1_0.NXM_OF_ETH_SRC, ofproto_v1_0.NXM_OF_ETH_SRC_W])
@MFField.register_field_header([ofproto_v1_0.NXM_OF_ETH_SRC, ofproto_v1_0.NXM_OF_ETH_SRC_W])
class MFEthSrc(MFField):
    pack_str = MF_PACK_STRING_MAC

    def __init__(self, header, value, mask=None):
        super(MFEthSrc, self).__init__(header, MFEthSrc.pack_str)
        self.value = value

    @classmethod
    def make(cls, header):
        return cls(header, MFEthSrc.pack_str)

    def put(self, buf, offset, rule):
        if rule.wc.dl_src_mask:
            return self.putw(buf, offset, rule.flow.dl_src, rule.wc.dl_src_mask)
        else:
            return self._put(buf, offset, rule.flow.dl_src)