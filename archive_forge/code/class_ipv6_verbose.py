import struct as _struct
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class ipv6_verbose(ipv6_compact):
    """An IPv6 dialect class - extra wide 'all zeroes' form."""
    word_fmt = '%.4x'
    compact = False