from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import re
import warnings
from passlib import hash
from passlib.utils import repeat_string
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, SkipTest
from passlib.tests.test_handlers import UPASS_USD, UPASS_TABLE
from passlib.tests.test_ext_django import DJANGO_VERSION, MIN_DJANGO_VERSION, \
from passlib.tests.test_handlers_argon2 import _base_argon2_test
class django_disabled_test(HandlerCase):
    """test django_disabled"""
    handler = hash.django_disabled
    disabled_contains_salt = True
    known_correct_hashes = [('password', '!'), ('', '!'), (UPASS_TABLE, '!')]
    known_alternate_hashes = [('!9wa845vn7098ythaehasldkfj', 'password', '!')]