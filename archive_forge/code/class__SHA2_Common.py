import hashlib
import logging; log = logging.getLogger(__name__)
from passlib.utils import safe_crypt, test_crypt, \
from passlib.utils.binary import h64
from passlib.utils.compat import byte_elem_value, u, \
import passlib.utils.handlers as uh
class _SHA2_Common(uh.HasManyBackends, uh.HasRounds, uh.HasSalt, uh.GenericHandler):
    """class containing common code shared by sha256_crypt & sha512_crypt"""
    setting_kwds = ('salt', 'rounds', 'implicit_rounds', 'salt_size')
    checksum_chars = uh.HASH64_CHARS
    max_salt_size = 16
    salt_chars = uh.HASH64_CHARS
    min_rounds = 1000
    max_rounds = 999999999
    rounds_cost = 'linear'
    _cdb_use_512 = False
    _rounds_prefix = None
    implicit_rounds = False

    def __init__(self, implicit_rounds=None, **kwds):
        super(_SHA2_Common, self).__init__(**kwds)
        if implicit_rounds is None:
            implicit_rounds = self.use_defaults and self.rounds == 5000
        self.implicit_rounds = implicit_rounds

    def _parse_salt(self, salt):
        return self._norm_salt(salt, relaxed=self.checksum is None)

    def _parse_rounds(self, rounds):
        return self._norm_rounds(rounds, relaxed=self.checksum is None)

    @classmethod
    def from_string(cls, hash):
        hash = to_unicode(hash, 'ascii', 'hash')
        ident = cls.ident
        if not hash.startswith(ident):
            raise uh.exc.InvalidHashError(cls)
        assert len(ident) == 3
        parts = hash[3:].split(_UDOLLAR)
        if parts[0].startswith(_UROUNDS):
            assert len(_UROUNDS) == 7
            rounds = parts.pop(0)[7:]
            if rounds.startswith(_UZERO) and rounds != _UZERO:
                raise uh.exc.ZeroPaddedRoundsError(cls)
            rounds = int(rounds)
            implicit_rounds = False
        else:
            rounds = 5000
            implicit_rounds = True
        if len(parts) == 2:
            salt, chk = parts
        elif len(parts) == 1:
            salt = parts[0]
            chk = None
        else:
            raise uh.exc.MalformedHashError(cls)
        return cls(rounds=rounds, salt=salt, checksum=chk or None, implicit_rounds=implicit_rounds)

    def to_string(self):
        if self.rounds == 5000 and self.implicit_rounds:
            hash = u('%s%s$%s') % (self.ident, self.salt, self.checksum or u(''))
        else:
            hash = u('%srounds=%d$%s$%s') % (self.ident, self.rounds, self.salt, self.checksum or u(''))
        return uascii_to_str(hash)
    backends = ('os_crypt', 'builtin')
    _test_hash = None

    @classmethod
    def _load_backend_os_crypt(cls):
        if test_crypt(*cls._test_hash):
            cls._set_calc_checksum_backend(cls._calc_checksum_os_crypt)
            return True
        else:
            return False

    def _calc_checksum_os_crypt(self, secret):
        config = self.to_string()
        hash = safe_crypt(secret, config)
        if hash is None:
            return self._calc_checksum_builtin(secret)
        cs = self.checksum_size
        if not hash.startswith(self.ident) or hash[-cs - 1] != _UDOLLAR:
            raise uh.exc.CryptBackendError(self, config, hash)
        return hash[-cs:]

    @classmethod
    def _load_backend_builtin(cls):
        cls._set_calc_checksum_backend(cls._calc_checksum_builtin)
        return True

    def _calc_checksum_builtin(self, secret):
        return _raw_sha2_crypt(secret, self.salt, self.rounds, self._cdb_use_512)