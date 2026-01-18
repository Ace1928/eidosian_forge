from base64 import b64encode, b64decode
from hashlib import md5, sha1, sha256, sha512
import logging; log = logging.getLogger(__name__)
import re
from passlib.handlers.misc import plaintext
from passlib.utils import unix_crypt_schemes, to_unicode
from passlib.utils.compat import uascii_to_str, unicode, u
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
class _SaltedBase64DigestHelper(uh.HasRawSalt, uh.HasRawChecksum, uh.GenericHandler):
    """helper for ldap_salted_md5 / ldap_salted_sha1"""
    setting_kwds = ('salt', 'salt_size')
    checksum_chars = uh.PADDED_BASE64_CHARS
    ident = None
    _hash_func = None
    _hash_regex = None
    min_salt_size = max_salt_size = 4
    min_salt_size = 4
    default_salt_size = 4
    max_salt_size = 16

    @classmethod
    def from_string(cls, hash):
        hash = to_unicode(hash, 'ascii', 'hash')
        m = cls._hash_regex.match(hash)
        if not m:
            raise uh.exc.InvalidHashError(cls)
        try:
            data = b64decode(m.group('tmp').encode('ascii'))
        except TypeError:
            raise uh.exc.MalformedHashError(cls)
        cs = cls.checksum_size
        assert cs
        return cls(checksum=data[:cs], salt=data[cs:])

    def to_string(self):
        data = self.checksum + self.salt
        hash = self.ident + b64encode(data).decode('ascii')
        return uascii_to_str(hash)

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        return self._hash_func(secret + self.salt).digest()