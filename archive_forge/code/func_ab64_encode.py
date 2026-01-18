from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def ab64_encode(data):
    """
    encode using shortened base64 format which omits padding & whitespace.
    uses custom ``./`` altchars.

    it is primarily used by Passlib's custom pbkdf2 hashes.
    """
    return b64s_encode(data).replace(b'+', b'.')