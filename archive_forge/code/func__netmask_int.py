import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
@property
def _netmask_int(self):
    """Same as self.netmask, but in integer format"""
    return self._module.max_int ^ self._hostmask_int