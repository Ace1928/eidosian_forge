from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class hex_sha1_test(HandlerCase):
    handler = hash.hex_sha1
    known_correct_hashes = [('password', '5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8'), (UPASS_TABLE, 'e059b2628e3a3e2de095679de9822c1d1466e0f0')]