import sys as _sys
from netaddr.core import (
from netaddr.strategy import ipv4 as _ipv4, ipv6 as _ipv6
def _set_prefixlen(self, value):
    if not isinstance(value, int):
        raise TypeError('int argument expected, not %s' % type(value))
    if not 0 <= value <= self._module.width:
        raise AddrFormatError('invalid prefix for an %s address!' % self._module.family_name)
    self._prefixlen = value