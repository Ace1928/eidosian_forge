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
class django_salted_sha1_test(HandlerCase, _DjangoHelper):
    """test django_salted_sha1"""
    handler = hash.django_salted_sha1
    max_django_version = (1, 9)
    known_correct_hashes = [('password', 'sha1$123abcdef$e4a1877b0e35c47329e7ed7e58014276168a37ba'), ('test', 'sha1$bcwHF9Hy8lxS$6b4cfa0651b43161c6f1471ce9523acf1f751ba3'), (UPASS_USD, 'sha1$c2e86$0f75c5d7fbd100d587c127ef0b693cde611b4ada'), (UPASS_TABLE, 'sha1$6d853$ef13a4d8fb57aed0cb573fe9c82e28dc7fd372d4'), ('MyPassword', 'sha1$54123$893cf12e134c3c215f3a76bd50d13f92404a54d3')]
    known_unidentified_hashes = ['md5$aa$bb']
    known_malformed_hashes = ['sha1$c2e86$0f75']
    FuzzHashGenerator = django_salted_md5_test.FuzzHashGenerator