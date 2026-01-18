import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class eui64_unix_expanded(eui64_unix):
    """A UNIX-style MAC address dialect class with leading zeroes."""
    word_fmt = '%.2x'