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
def _parse_cookie(data):
    if not PY2:
        data = data.encode('latin-1')
    for key, val in _rx_cookie.findall(data):
        yield (key, _unquote(val))