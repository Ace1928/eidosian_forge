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
def _coerce_vary_rounds(value):
    """parse vary_rounds string to percent as [0,1) float, or integer"""
    if value.endswith('%'):
        return float(value.rstrip('%')) * 0.01
    try:
        return int(value)
    except ValueError:
        return float(value)