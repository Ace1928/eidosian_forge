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
@_set_nxm_headers([ofproto_v1_0.nxm_nx_reg(i) for i in range(FLOW_N_REGS)] + [ofproto_v1_0.nxm_nx_reg_w(i) for i in range(FLOW_N_REGS)])
class MFRegister(MFField):

    @classmethod
    def make(cls, header):
        return cls(header, MF_PACK_STRING_BE32)

    def put(self, buf, offset, rule):
        for i in range(FLOW_N_REGS):
            if ofproto_v1_0.nxm_nx_reg(i) == self.nxm_header or ofproto_v1_0.nxm_nx_reg_w(i) == self.nxm_header:
                if rule.wc.regs_mask[i]:
                    return self.putm(buf, offset, rule.flow.regs[i], rule.wc.regs_mask[i])
                else:
                    return self._put(buf, offset, rule.flow.regs[i])