import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def _is_all_zero_bit(self, type_, val):
    if type_ == 'int' or type_ == 'ipv4':
        return val == 0
    elif type_ == 'mac':
        for v in val:
            if v != b'\x00':
                return False
        return True
    elif type_ == 'ipv6':
        for v in val:
            if v != 0:
                return False
        return True
    else:
        raise Exception('Unknown type')