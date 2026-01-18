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
class _PureBackend(_Argon2Common):
    """
    argon2pure backend
    """

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        assert mixin_cls is _PureBackend
        global _argon2pure
        try:
            import argon2pure as _argon2pure
        except ImportError:
            return False
        try:
            from argon2pure import ARGON2_DEFAULT_VERSION as max_version
        except ImportError:
            log.warning("detected 'argon2pure' backend, but package is too old (passlib requires argon2pure >= 1.2.3)")
            return False
        log.debug("detected 'argon2pure' backend, with support for 0x%x argon2 hashes", max_version)
        if not dryrun:
            warn("Using argon2pure backend, which is 100x+ slower than is required for adequate security. Installing argon2_cffi (via 'pip install argon2_cffi') is strongly recommended", exc.PasslibSecurityWarning)
        type_map = {}
        for type in ALL_TYPES:
            try:
                type_map[type] = getattr(_argon2pure, 'ARGON2' + type.upper())
            except AttributeError:
                assert type not in (TYPE_I, TYPE_D), 'unexpected missing type: %r' % type
        mixin_cls._backend_type_map = type_map
        mixin_cls.version = mixin_cls.max_version = max_version
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    def _calc_checksum(self, secret):
        uh.validate_secret(secret)
        secret = to_bytes(secret, 'utf-8')
        kwds = dict(password=secret, salt=self.salt, time_cost=self.rounds, memory_cost=self.memory_cost, parallelism=self.parallelism, tag_length=self.checksum_size, type_code=self._get_backend_type(self.type), version=self.version)
        if self.max_threads > 0:
            kwds['threads'] = self.max_threads
        if self.pure_use_threads:
            kwds['use_threads'] = True
        if self.data:
            kwds['associated_data'] = self.data
        try:
            return _argon2pure.argon2(**kwds)
        except _argon2pure.Argon2Error as err:
            raise self._adapt_backend_error(err, self=self)