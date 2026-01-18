import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class mac_pgsql(mac_eui48):
    """A PostgreSQL style (2 x 24-bit words) MAC address dialect class."""
    word_size = 24
    num_words = width // word_size
    word_sep = ':'
    word_fmt = '%.6x'
    word_base = 16