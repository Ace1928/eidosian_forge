from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
@classmethod
def _norm_type(cls, value):
    if not isinstance(value, unicode):
        if PY2 and isinstance(value, bytes):
            value = value.decode('ascii')
        else:
            raise uh.exc.ExpectedTypeError(value, 'str', 'type')
    if value in ALL_TYPES_SET:
        return value
    temp = value.lower()
    if temp in ALL_TYPES_SET:
        return temp
    raise ValueError('unknown argon2 hash type: %r' % (value,))