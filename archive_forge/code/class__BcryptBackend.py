from __future__ import with_statement, absolute_import
from base64 import b64encode
from hashlib import sha256
import os
import re
import logging; log = logging.getLogger(__name__)
from warnings import warn
from passlib.crypto.digest import compile_hmac
from passlib.exc import PasslibHashWarning, PasslibSecurityWarning, PasslibSecurityError
from passlib.utils import safe_crypt, repeat_string, to_bytes, parse_version, \
from passlib.utils.binary import bcrypt64
from passlib.utils.compat import get_unbound_method_function
from passlib.utils.compat import u, uascii_to_str, unicode, str_to_uascii, PY3, error_from
import passlib.utils.handlers as uh
class _BcryptBackend(_BcryptCommon):
    """
    backend which uses 'bcrypt' package
    """

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        global _bcrypt
        if _detect_pybcrypt():
            return False
        try:
            import bcrypt as _bcrypt
        except ImportError:
            return False
        try:
            version = _bcrypt.__about__.__version__
        except:
            log.warning('(trapped) error reading bcrypt version', exc_info=True)
            version = '<unknown>'
        log.debug("detected 'bcrypt' backend, version %r", version)
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    def _calc_checksum(self, secret):
        secret, ident = self._prepare_digest_args(secret)
        config = self._get_config(ident)
        if isinstance(config, unicode):
            config = config.encode('ascii')
        hash = _bcrypt.hashpw(secret, config)
        assert isinstance(hash, bytes)
        if not hash.startswith(config) or len(hash) != len(config) + 31:
            raise uh.exc.CryptBackendError(self, config, hash, source='`bcrypt` package')
        return hash[-31:].decode('ascii')