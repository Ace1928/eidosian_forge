from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
class TotpToken(SequenceMixin):
    """
    Object returned by :meth:`TOTP.generate`.
    It can be treated as a sequence of ``(token, expire_time)``,
    or accessed via the following attributes:

    .. autoattribute:: token
    .. autoattribute:: expire_time
    .. autoattribute:: counter
    .. autoattribute:: remaining
    .. autoattribute:: valid
    """
    totp = None
    token = None
    counter = None

    def __init__(self, totp, token, counter):
        """
        .. warning::
            the constructor signature is an internal detail, and is subject to change.
        """
        self.totp = totp
        self.token = token
        self.counter = counter

    @memoized_property
    def start_time(self):
        """Timestamp marking beginning of period when token is valid"""
        return self.totp._counter_to_time(self.counter)

    @memoized_property
    def expire_time(self):
        """Timestamp marking end of period when token is valid"""
        return self.totp._counter_to_time(self.counter + 1)

    @property
    def remaining(self):
        """number of (float) seconds before token expires"""
        return max(0, self.expire_time - self.totp.now())

    @property
    def valid(self):
        """whether token is still valid"""
        return bool(self.remaining)

    def _as_tuple(self):
        return (self.token, self.expire_time)

    def __repr__(self):
        expired = '' if self.remaining else ' expired'
        return "<TotpToken token='%s' expire_time=%d%s>" % (self.token, self.expire_time, expired)