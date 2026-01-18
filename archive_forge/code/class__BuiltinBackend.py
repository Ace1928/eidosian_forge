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
class _BuiltinBackend(_BcryptCommon):
    """
    backend which uses passlib's pure-python implementation
    """

    @classmethod
    def _load_backend_mixin(mixin_cls, name, dryrun):
        from passlib.utils import as_bool
        if not as_bool(os.environ.get('PASSLIB_BUILTIN_BCRYPT')):
            log.debug("bcrypt 'builtin' backend not enabled via $PASSLIB_BUILTIN_BCRYPT")
            return False
        global _builtin_bcrypt
        from passlib.crypto._blowfish import raw_bcrypt as _builtin_bcrypt
        return mixin_cls._finalize_backend_mixin(name, dryrun)

    def _calc_checksum(self, secret):
        secret, ident = self._prepare_digest_args(secret)
        chk = _builtin_bcrypt(secret, ident[1:-1], self.salt.encode('ascii'), self.rounds)
        return chk.decode('ascii')