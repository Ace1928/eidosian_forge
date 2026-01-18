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
class _PyBcryptBackend(_BcryptCommon):
    """
    backend which uses 'pybcrypt' package
    """
    _calc_lock = None

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        global _pybcrypt
        if not _detect_pybcrypt():
            return False
        try:
            import bcrypt as _pybcrypt
        except ImportError:
            return False
        if not dryrun:
            warn('Support for `py-bcrypt` is deprecated, and will be removed in Passlib 1.8; Please use `pip install bcrypt` instead', DeprecationWarning)
        try:
            version = _pybcrypt._bcrypt.__version__
        except:
            log.warning('(trapped) error reading pybcrypt version', exc_info=True)
            version = '<unknown>'
        log.debug("detected 'pybcrypt' backend, version %r", version)
        vinfo = parse_version(version) or (0, 0)
        if vinfo < (0, 3):
            warn('py-bcrypt %s has a major security vulnerability, you should upgrade to py-bcrypt 0.3 immediately.' % version, uh.exc.PasslibSecurityWarning)
            if mixin_cls._calc_lock is None:
                import threading
                mixin_cls._calc_lock = threading.Lock()
            mixin_cls._calc_checksum = get_unbound_method_function(mixin_cls._calc_checksum_threadsafe)
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    def _calc_checksum_threadsafe(self, secret):
        with self._calc_lock:
            return self._calc_checksum_raw(secret)

    def _calc_checksum_raw(self, secret):
        secret, ident = self._prepare_digest_args(secret)
        config = self._get_config(ident)
        hash = _pybcrypt.hashpw(secret, config)
        if not hash.startswith(config) or len(hash) != len(config) + 31:
            raise uh.exc.CryptBackendError(self, config, hash, source='pybcrypt library')
        return str_to_uascii(hash[-31:])
    _calc_checksum = _calc_checksum_raw