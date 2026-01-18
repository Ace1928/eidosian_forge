from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
class _CffiBackend(_Argon2Common):
    """
    argon2_cffi backend
    """

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        assert mixin_cls is _CffiBackend
        if _argon2_cffi is None:
            if _argon2_cffi_error:
                raise exc.PasslibSecurityError(_argon2_cffi_error)
            return False
        max_version = _argon2_cffi.low_level.ARGON2_VERSION
        log.debug("detected 'argon2_cffi' backend, version %r, with support for 0x%x argon2 hashes", _argon2_cffi.__version__, max_version)
        TypeEnum = _argon2_cffi.Type
        type_map = {}
        for type in ALL_TYPES:
            try:
                type_map[type] = getattr(TypeEnum, type.upper())
            except AttributeError:
                assert type not in (TYPE_I, TYPE_D), 'unexpected missing type: %r' % type
        mixin_cls._backend_type_map = type_map
        mixin_cls.version = mixin_cls.max_version = max_version
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    @classmethod
    def hash(cls, secret):
        uh.validate_secret(secret)
        secret = to_bytes(secret, 'utf-8')
        try:
            return bascii_to_str(_argon2_cffi.low_level.hash_secret(type=cls._get_backend_type(cls.type), memory_cost=cls.memory_cost, time_cost=cls.default_rounds, parallelism=cls.parallelism, salt=to_bytes(cls._generate_salt()), hash_len=cls.checksum_size, secret=secret))
        except _argon2_cffi.exceptions.HashingError as err:
            raise cls._adapt_backend_error(err)
    _byte_ident_map = dict(((render_bytes(b'$argon2%s$', type.encode('ascii')), type) for type in ALL_TYPES))

    @classmethod
    def verify(cls, secret, hash):
        uh.validate_secret(secret)
        secret = to_bytes(secret, 'utf-8')
        hash = to_bytes(hash, 'ascii')
        type = cls._byte_ident_map.get(hash[:1 + hash.find(b'$', 1)], TYPE_I)
        type_code = cls._get_backend_type(type)
        try:
            result = _argon2_cffi.low_level.verify_secret(hash, secret, type_code)
            assert result is True
            return True
        except _argon2_cffi.exceptions.VerifyMismatchError:
            return False
        except _argon2_cffi.exceptions.VerificationError as err:
            raise cls._adapt_backend_error(err, hash=hash)

    @classmethod
    def genhash(cls, secret, config):
        uh.validate_secret(secret)
        secret = to_bytes(secret, 'utf-8')
        self = cls.from_string(config)
        try:
            result = bascii_to_str(_argon2_cffi.low_level.hash_secret(type=cls._get_backend_type(self.type), memory_cost=self.memory_cost, time_cost=self.rounds, parallelism=self.parallelism, salt=to_bytes(self.salt), hash_len=self.checksum_size, secret=secret, version=self.version))
        except _argon2_cffi.exceptions.HashingError as err:
            raise cls._adapt_backend_error(err, hash=config)
        if self.version == 16:
            result = result.replace('$v=16$', '$')
        return result

    def _calc_checksum(self, secret):
        raise AssertionError("shouldn't be called under argon2_cffi backend")