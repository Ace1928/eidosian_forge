from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
@staticmethod
def _parse_config_key(ckey):
    """helper used to parse ``cat__scheme__option`` keys into a tuple"""
    assert isinstance(ckey, native_string_types)
    parts = ckey.replace('.', '__').split('__')
    count = len(parts)
    if count == 1:
        cat, scheme, key = (None, None, parts[0])
    elif count == 2:
        cat = None
        scheme, key = parts
    elif count == 3:
        cat, scheme, key = parts
    else:
        raise TypeError('keys must have less than 3 separators: %r' % (ckey,))
    if cat == 'default':
        cat = None
    elif not cat and cat is not None:
        raise TypeError('empty category: %r' % ckey)
    if scheme == 'context':
        scheme = None
    elif not scheme and scheme is not None:
        raise TypeError('empty scheme: %r' % ckey)
    if not key:
        raise TypeError('empty option: %r' % ckey)
    return (cat, scheme, key)