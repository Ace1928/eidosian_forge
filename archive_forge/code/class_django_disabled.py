from base64 import b64encode
from binascii import hexlify
from hashlib import md5, sha1, sha256
import logging; log = logging.getLogger(__name__)
from passlib.handlers.bcrypt import _wrapped_bcrypt
from passlib.hash import argon2, bcrypt, pbkdf2_sha1, pbkdf2_sha256
from passlib.utils import to_unicode, rng, getrandstr
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import str_to_uascii, uascii_to_str, unicode, u
from passlib.crypto.digest import pbkdf2_hmac
import passlib.utils.handlers as uh
class django_disabled(uh.ifc.DisabledHash, uh.StaticHandler):
    """This class provides disabled password behavior for Django, and follows the :ref:`password-hash-api`.

    This class does not implement a hash, but instead
    claims the special hash string ``"!"`` which Django uses
    to indicate an account's password has been disabled.

    * newly encrypted passwords will hash to ``"!"``.
    * it rejects all passwords.

    .. note::

        Django 1.6 prepends a randomly generated 40-char alphanumeric string
        to each unusuable password. This class recognizes such strings,
        but for backwards compatibility, still returns ``"!"``.

        See `<https://code.djangoproject.com/ticket/20079>`_ for why
        Django appends an alphanumeric string.

    .. versionchanged:: 1.6.2 added Django 1.6 support

    .. versionchanged:: 1.7 started appending an alphanumeric string.
    """
    name = 'django_disabled'
    _hash_prefix = u('!')
    suffix_length = 40

    @classmethod
    def identify(cls, hash):
        hash = uh.to_unicode_for_identify(hash)
        return hash.startswith(cls._hash_prefix)

    def _calc_checksum(self, secret):
        return getrandstr(rng, BASE64_CHARS[:-2], self.suffix_length)

    @classmethod
    def verify(cls, secret, hash):
        uh.validate_secret(secret)
        if not cls.identify(hash):
            raise uh.exc.InvalidHashError(cls)
        return False