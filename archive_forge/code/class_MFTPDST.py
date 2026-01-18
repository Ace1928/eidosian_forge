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
@_set_nxm_headers([ofproto_v1_0.NXM_OF_TCP_DST, ofproto_v1_0.NXM_OF_TCP_DST_W, ofproto_v1_0.NXM_OF_UDP_DST, ofproto_v1_0.NXM_OF_UDP_DST_W])
class MFTPDST(MFField):

    @classmethod
    def make(cls, header):
        return cls(header, MF_PACK_STRING_BE16)

    def put(self, buf, offset, rule):
        return self.putm(buf, offset, rule.flow.tp_dst, rule.wc.tp_dst_mask)