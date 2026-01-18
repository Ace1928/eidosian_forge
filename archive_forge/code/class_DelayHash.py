from __future__ import with_statement
from passlib.utils.compat import PY3
import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.context import CryptContext, LazyCryptContext
from passlib.exc import PasslibConfigWarning, PasslibHashWarning
from passlib.utils import tick, to_unicode
from passlib.utils.compat import irange, u, unicode, str_to_uascii, PY2, PY26
import passlib.utils.handlers as uh
from passlib.tests.utils import (TestCase, set_file, TICK_RESOLUTION,
from passlib.registry import (register_crypt_handler_path,
import hashlib, time
class DelayHash(uh.StaticHandler):
    """dummy hasher which delays by specified amount"""
    name = 'delay_hash'
    checksum_chars = uh.LOWER_HEX_CHARS
    checksum_size = 40
    delay = 0
    _hash_prefix = u('$x$')

    def _calc_checksum(self, secret):
        time.sleep(self.delay)
        if isinstance(secret, unicode):
            secret = secret.encode('utf-8')
        return str_to_uascii(hashlib.sha1(b'prefix' + secret).hexdigest())