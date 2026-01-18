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
class _OsCryptBackend(_BcryptCommon):
    """
    backend which uses :func:`crypt.crypt`
    """
    _require_valid_utf8_bytes = not crypt_accepts_bytes

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        if not test_crypt('test', TEST_HASH_2A):
            return False
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    def _calc_checksum(self, secret):
        secret, ident = self._prepare_digest_args(secret)
        config = self._get_config(ident)
        hash = safe_crypt(secret, config)
        if hash is not None:
            if not hash.startswith(config) or len(hash) != len(config) + 31:
                raise uh.exc.CryptBackendError(self, config, hash)
            return hash[-31:]
        if PY3 and isinstance(secret, bytes):
            try:
                secret.decode('utf-8')
            except UnicodeDecodeError:
                raise error_from(uh.exc.PasswordValueError('python3 crypt.crypt() ony supports bytes passwords using UTF8; passlib recommends running `pip install bcrypt` for general bcrypt support.'), None)
        debug_only_repr = uh.exc.debug_only_repr
        raise uh.exc.InternalBackendError('crypt.crypt() failed for unknown reason; passlib recommends running `pip install bcrypt` for general bcrypt support.(config=%s, secret=%s)' % (debug_only_repr(config), debug_only_repr(secret)))