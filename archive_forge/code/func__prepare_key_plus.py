from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def _prepare_key_plus(alg, keystr):
    if isinstance(keystr, bytes):
        keystr = keystr.decode('utf-8')
    return alg.prepare_key(keystr)