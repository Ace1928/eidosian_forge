from binascii import hexlify, unhexlify
from hashlib import sha1
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_unicode, xor_bytes
from passlib.utils.compat import irange, u, \
from passlib.crypto.des import des_encrypt_block
import passlib.utils.handlers as uh
class oracle10(uh.HasUserContext, uh.StaticHandler):
    """This class implements the password hash used by Oracle up to version 10g, and follows the :ref:`password-hash-api`.

    It does a single round of hashing, and relies on the username as the salt.

    The :meth:`~passlib.ifc.PasswordHash.hash`, :meth:`~passlib.ifc.PasswordHash.genhash`, and :meth:`~passlib.ifc.PasswordHash.verify` methods all require the
    following additional contextual keywords:

    :type user: str
    :param user: name of oracle user account this password is associated with.
    """
    name = 'oracle10'
    checksum_chars = uh.HEX_CHARS
    checksum_size = 16

    @classmethod
    def _norm_hash(cls, hash):
        return hash.upper()

    def _calc_checksum(self, secret):
        if isinstance(secret, bytes):
            secret = secret.decode('utf-8')
        user = to_unicode(self.user, 'utf-8', param='user')
        input = (user + secret).upper().encode('utf-16-be')
        hash = des_cbc_encrypt(ORACLE10_MAGIC, input)
        hash = des_cbc_encrypt(hash, input)
        return hexlify(hash).decode('ascii').upper()