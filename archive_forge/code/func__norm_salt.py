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
@classmethod
def _norm_salt(cls, salt, **kwds):
    salt = super(_BcryptCommon, cls)._norm_salt(salt, **kwds)
    assert salt is not None, "HasSalt didn't generate new salt!"
    changed, salt = bcrypt64.check_repair_unused(salt)
    if changed:
        warn('encountered a bcrypt salt with incorrectly set padding bits; you may want to use bcrypt.normhash() to fix this; this will be an error under Passlib 2.0', PasslibHashWarning)
    return salt