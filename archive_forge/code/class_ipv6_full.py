import struct as _struct
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class ipv6_full(ipv6_compact):
    """An IPv6 dialect class - 'all zeroes' form."""
    compact = False