from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
def _encode_user(self, user):
    """user-specific wrapper for _encode_field()"""
    return self._encode_field(user, 'user')