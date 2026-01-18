import base64
import binascii
import hashlib
import hmac
import json
from datetime import (
import re
import string
import time
import warnings
from webob.compat import (
from webob.util import strings_differ
def _value_quote(v):
    leftovers = v.translate(None, _allowed_cookie_bytes)
    if leftovers:
        __warn_or_raise('Cookie value contains invalid bytes: (%r). Future versions will raise ValueError upon encountering invalid bytes.' % (leftovers,), RuntimeWarning, ValueError, 'Invalid characters in cookie value')
        return b'"' + b''.join(map(_escape_char, v)) + b'"'
    return v