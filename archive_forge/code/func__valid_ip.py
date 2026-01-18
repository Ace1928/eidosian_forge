import numbers
import struct
import netaddr
from os_ken.lib import addrconv
from os_ken.lib import type_desc
def _valid_ip(strategy, bits, addr, flags=0):
    addr = addr.split('/')
    if len(addr) == 1:
        return strategy(addr[0], flags)
    elif len(addr) == 2:
        return strategy(addr[0], flags) and 0 <= int(addr[1]) <= bits
    else:
        return False