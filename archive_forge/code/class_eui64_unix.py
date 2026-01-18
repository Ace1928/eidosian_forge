import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class eui64_unix(eui64_base):
    """A UNIX-style MAC address dialect class."""
    word_size = 8
    num_words = width // word_size
    word_sep = ':'
    word_fmt = '%x'
    word_base = 16