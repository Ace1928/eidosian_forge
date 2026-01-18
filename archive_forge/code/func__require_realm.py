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
def _require_realm(self, realm):
    if realm is None:
        realm = self.default_realm
        if realm is None:
            raise TypeError('you must specify a realm explicitly, or set the default_realm attribute')
    return realm