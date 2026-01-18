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
def _render_config_key(key):
    """convert 3-part config key to single string"""
    cat, scheme, option = key
    if cat:
        return '%s__%s__%s' % (cat, scheme or 'context', option)
    elif scheme:
        return '%s__%s' % (scheme, option)
    else:
        return option