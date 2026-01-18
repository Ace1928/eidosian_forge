from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class plaintext_test(HandlerCase):
    handler = hash.plaintext
    accepts_all_hashes = True
    known_correct_hashes = [('', ''), ('password', 'password'), (UPASS_TABLE, UPASS_TABLE if PY3 else PASS_TABLE_UTF8), (PASS_TABLE_UTF8, UPASS_TABLE if PY3 else PASS_TABLE_UTF8)]