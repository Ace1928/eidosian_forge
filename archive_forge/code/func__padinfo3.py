from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
@memoized_property
def _padinfo3(self):
    """mask to clear padding bits, and valid last bytes (for strings 3 % 4)"""
    bits = 3 if self.big else 3 << 4
    return (~bits, self.__make_padset(bits))