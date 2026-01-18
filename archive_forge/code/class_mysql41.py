from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.utils import to_native_str
from passlib.utils.compat import bascii_to_str, unicode, u, \
import passlib.utils.handlers as uh
class mysql41(uh.StaticHandler):
    """This class implements the MySQL 4.1 password hash, and follows the :ref:`password-hash-api`.

    It has no salt and a single fixed round.

    The :meth:`~passlib.ifc.PasswordHash.hash` and :meth:`~passlib.ifc.PasswordHash.genconfig` methods accept no optional keywords.
    """
    name = 'mysql41'
    _hash_prefix = u('*')
    checksum_chars = uh.HEX_CHARS
    checksum_size = 40

    @classmethod
    def _norm_hash(cls, hash):
        return hash.upper()

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        return str_to_uascii(sha1(sha1(secret).digest()).hexdigest()).upper()