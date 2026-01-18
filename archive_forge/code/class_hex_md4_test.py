from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class hex_md4_test(HandlerCase):
    handler = hash.hex_md4
    known_correct_hashes = [('password', '8a9d093f14f8701df17732b2bb182c74'), (UPASS_TABLE, '876078368c47817ce5f9115f3a42cf74')]