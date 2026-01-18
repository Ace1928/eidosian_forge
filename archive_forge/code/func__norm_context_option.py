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
def _norm_context_option(self, cat, key, value):
    schemes = self.schemes
    if key == 'default':
        if hasattr(value, 'name'):
            value = value.name
        elif not isinstance(value, native_string_types):
            raise ExpectedTypeError(value, 'str', 'default')
        if schemes and value not in schemes:
            raise KeyError('default scheme not found in policy')
    elif key == 'deprecated':
        if isinstance(value, native_string_types):
            value = splitcomma(value)
        elif not isinstance(value, (list, tuple)):
            raise ExpectedTypeError(value, 'str or seq', 'deprecated')
        if 'auto' in value:
            if len(value) > 1:
                raise ValueError("cannot list other schemes if ``deprecated=['auto']`` is used")
        elif schemes:
            for scheme in value:
                if not isinstance(scheme, native_string_types):
                    raise ExpectedTypeError(value, 'str', 'deprecated element')
                if scheme not in schemes:
                    raise KeyError('deprecated scheme not found in policy: %r' % (scheme,))
    elif key == 'min_verify_time':
        warn("'min_verify_time' was deprecated in Passlib 1.6, is ignored in 1.7, and will be removed in 1.8", DeprecationWarning)
    elif key == 'harden_verify':
        warn("'harden_verify' is deprecated & ignored as of Passlib 1.7.1,  and will be removed in 1.8", DeprecationWarning)
    elif key != 'schemes':
        raise KeyError('unknown CryptContext keyword: %r' % (key,))
    return (key, value)