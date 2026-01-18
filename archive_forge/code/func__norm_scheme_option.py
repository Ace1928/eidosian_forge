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
def _norm_scheme_option(self, key, value):
    if key in _forbidden_scheme_options:
        raise KeyError('%r option not allowed in CryptContext configuration' % (key,))
    if isinstance(value, native_string_types):
        func = _coerce_scheme_options.get(key)
        if func:
            value = func(value)
    return (key, value)