import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import safe_crypt, test_crypt, to_unicode
from passlib.utils.binary import h64, h64big
from passlib.utils.compat import byte_elem_value, u, uascii_to_str, unicode, suppress_cause
from passlib.crypto.des import des_encrypt_int_block
import passlib.utils.handlers as uh
def _crypt_secret_to_key(secret):
    """convert secret to 64-bit DES key.

    this only uses the first 8 bytes of the secret,
    and discards the high 8th bit of each byte at that.
    a null parity bit is inserted after every 7th bit of the output.
    """
    return sum(((byte_elem_value(c) & 127) << 57 - i * 8 for i, c in enumerate(secret[:8])))