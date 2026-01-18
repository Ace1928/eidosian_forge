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
def _detect_pybcrypt():
    """
    internal helper which tries to distinguish pybcrypt vs bcrypt.

    :returns:
        True if cext-based py-bcrypt,
        False if ffi-based bcrypt,
        None if 'bcrypt' module not found.

    .. versionchanged:: 1.6.3

        Now assuming bcrypt installed, unless py-bcrypt explicitly detected.
        Previous releases assumed py-bcrypt by default.

        Making this change since py-bcrypt is (apparently) unmaintained and static,
        whereas bcrypt is being actively maintained, and it's internal structure may shift.
    """
    try:
        import bcrypt
    except ImportError:
        return None
    try:
        from bcrypt._bcrypt import __version__
    except ImportError:
        return False
    return True