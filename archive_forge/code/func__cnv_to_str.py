import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def _cnv_to_str(self, type_, value, mask, f_value, f_mask):
    func = None
    if type_ == 'int':
        pass
    elif type_ == 'mac':
        func = self.haddr_to_str
    elif type_ == 'ipv4':
        func = self.ipv4_to_str
    elif type_ == 'ipv6':
        func = self.ipv6_to_str
    else:
        raise Exception('Unknown type')
    if func:
        value = func(value)
        f_value = func(f_value)
        if mask:
            mask = func(mask)
        if f_mask:
            f_mask = func(f_mask)
    return (value, mask, f_value, f_mask)