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
class DjangoVariableHash(uh.HasRounds, DjangoSaltedHash):
    """base class providing common code for django hashes w/ variable rounds"""
    setting_kwds = DjangoSaltedHash.setting_kwds + ('rounds',)
    min_rounds = 1

    @classmethod
    def from_string(cls, hash):
        rounds, salt, chk = uh.parse_mc3(hash, cls.ident, handler=cls)
        return cls(rounds=rounds, salt=salt, checksum=chk)

    def to_string(self):
        return uh.render_mc3(self.ident, self.rounds, self.salt, self.checksum)