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

            check for bsd wraparound bug (fixed in 2b)
            this is treated as a warning, because it's rare in the field,
            and pybcrypt (as of 2015-7-21) is unpatched, but some people may be stuck with it.

            test cases from <http://www.openwall.com/lists/oss-security/2012/01/02/4>

            NOTE: reference hash is of password "0"*72

            NOTE: if in future we need to deliberately create hashes which have this bug,
                  can use something like 'hashpw(repeat_string(secret[:((1+secret) % 256) or 1]), 72)'
            