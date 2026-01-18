from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_native_str
from passlib.utils.compat import bascii_to_str, unicode, u, \
import passlib.utils.handlers as uh
class mysql323(uh.StaticHandler):
    """This class implements the MySQL 3.2.3 password hash, and follows the :ref:`password-hash-api`.

    It has no salt and a single fixed round.

    The :meth:`~passlib.ifc.PasswordHash.hash` and :meth:`~passlib.ifc.PasswordHash.genconfig` methods accept no optional keywords.
    """
    name = 'mysql323'
    checksum_size = 16
    checksum_chars = uh.HEX_CHARS

    @classmethod
    def _norm_hash(cls, hash):
        return hash.lower()

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        MASK_32 = 4294967295
        MASK_31 = 2147483647
        WHITE = b' \t'
        nr1 = 1345345333
        nr2 = 305419889
        add = 7
        for c in secret:
            if c in WHITE:
                continue
            tmp = byte_elem_value(c)
            nr1 ^= ((nr1 & 63) + add) * tmp + (nr1 << 8) & MASK_32
            nr2 = nr2 + (nr2 << 8 ^ nr1) & MASK_32
            add = add + tmp & MASK_32
        return u('%08x%08x') % (nr1 & MASK_31, nr2 & MASK_31)