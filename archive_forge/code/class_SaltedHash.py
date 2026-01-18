from __future__ import with_statement
import re
import hashlib
from logging import getLogger
import warnings
from passlib.hash import ldap_md5, sha256_crypt
from passlib.exc import MissingBackendError, PasslibHashWarning
from passlib.utils.compat import str_to_uascii, \
import passlib.utils.handlers as uh
from passlib.tests.utils import HandlerCase, TestCase
from passlib.utils.compat import u
class SaltedHash(uh.HasSalt, uh.GenericHandler):
    """test algorithm with a salt"""
    name = 'salted_test_hash'
    setting_kwds = ('salt',)
    min_salt_size = 2
    max_salt_size = 4
    checksum_size = 40
    salt_chars = checksum_chars = uh.LOWER_HEX_CHARS
    _hash_regex = re.compile(u('^@salt[0-9a-f]{42,44}$'))

    @classmethod
    def from_string(cls, hash):
        if not cls.identify(hash):
            raise uh.exc.InvalidHashError(cls)
        if isinstance(hash, bytes):
            hash = hash.decode('ascii')
        return cls(salt=hash[5:-40], checksum=hash[-40:])

    def to_string(self):
        hash = u('@salt%s%s') % (self.salt, self.checksum)
        return uascii_to_str(hash)

    def _calc_checksum(self, secret):
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        data = self.salt.encode('ascii') + secret + self.salt.encode('ascii')
        return str_to_uascii(hashlib.sha1(data).hexdigest())