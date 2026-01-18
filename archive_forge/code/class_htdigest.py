import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import to_native_str, to_bytes, render_bytes, consteq
from passlib.utils.compat import unicode, str_to_uascii
import passlib.utils.handlers as uh
from passlib.crypto.digest import lookup_hash
class htdigest(uh.MinimalHandler):
    """htdigest hash function.

    .. todo::
        document this hash
    """
    name = 'htdigest'
    setting_kwds = ()
    context_kwds = ('user', 'realm', 'encoding')
    default_encoding = 'utf-8'

    @classmethod
    def hash(cls, secret, user, realm, encoding=None):
        if not encoding:
            encoding = cls.default_encoding
        uh.validate_secret(secret)
        if isinstance(secret, unicode):
            secret = secret.encode(encoding)
        user = to_bytes(user, encoding, 'user')
        realm = to_bytes(realm, encoding, 'realm')
        data = render_bytes('%s:%s:%s', user, realm, secret)
        return hashlib.md5(data).hexdigest()

    @classmethod
    def _norm_hash(cls, hash):
        """normalize hash to native string, and validate it"""
        hash = to_native_str(hash, param='hash')
        if len(hash) != 32:
            raise uh.exc.MalformedHashError(cls, 'wrong size')
        for char in hash:
            if char not in uh.LC_HEX_CHARS:
                raise uh.exc.MalformedHashError(cls, 'invalid chars in hash')
        return hash

    @classmethod
    def verify(cls, secret, hash, user, realm, encoding='utf-8'):
        hash = cls._norm_hash(hash)
        other = cls.hash(secret, user, realm, encoding)
        return consteq(hash, other)

    @classmethod
    def identify(cls, hash):
        try:
            cls._norm_hash(hash)
        except ValueError:
            return False
        return True

    @uh.deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genconfig(cls):
        return cls.hash('', '', '')

    @uh.deprecated_method(deprecated='1.7', removed='2.0')
    @classmethod
    def genhash(cls, secret, config, user, realm, encoding=None):
        cls._norm_hash(config)
        return cls.hash(secret, user, realm, encoding)