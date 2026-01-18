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
def _valid_cookie_name(self, name):
    if not isinstance(name, string_types):
        raise TypeError(name, 'cookie name must be a string')
    if not isinstance(name, text_type):
        name = text_(name, 'utf-8')
    try:
        bytes_cookie_name = bytes_(name, 'ascii')
    except UnicodeEncodeError:
        raise TypeError('cookie name must be encodable to ascii')
    if not _valid_cookie_name(bytes_cookie_name):
        raise TypeError('cookie name must be valid according to RFC 6265')
    return name