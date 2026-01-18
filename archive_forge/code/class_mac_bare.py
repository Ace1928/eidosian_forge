import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class mac_bare(mac_eui48):
    """A bare (no delimiters) MAC address dialect class."""
    word_size = 48
    num_words = width // word_size
    word_sep = ''
    word_fmt = '%.12X'
    word_base = 16