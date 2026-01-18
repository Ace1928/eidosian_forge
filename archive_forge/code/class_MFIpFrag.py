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
@_set_nxm_headers([ofproto_v1_0.NXM_NX_IP_FRAG, ofproto_v1_0.NXM_NX_IP_FRAG_W])
class MFIpFrag(MFField):

    @classmethod
    def make(cls, header):
        return cls(header, '!B')

    def put(self, buf, offset, rule):
        if rule.wc.nw_frag_mask == FLOW_NW_FRAG_MASK:
            return self._put(buf, offset, rule.flow.nw_frag)
        else:
            return self.putw(buf, offset, rule.flow.nw_frag, rule.wc.nw_frag_mask & FLOW_NW_FRAG_MASK)