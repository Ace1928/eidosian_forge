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
class django_des_crypt_test(HandlerCase, _DjangoHelper):
    """test django_des_crypt"""
    handler = hash.django_des_crypt
    max_django_version = (1, 9)
    known_correct_hashes = [('password', 'crypt$c2$c2M87q...WWcU'), ('password', 'crypt$c2e86$c2M87q...WWcU'), ('passwordignoreme', 'crypt$c2.AZ$c2M87q...WWcU'), (UPASS_USD, 'crypt$c2e86$c2hN1Bxd6ZiWs'), (UPASS_TABLE, 'crypt$0.aQs$0.wB.TT0Czvlo'), (u('hellÖ'), 'crypt$sa$saykDgk3BPZ9E'), ('foo', 'crypt$MNVY.9ajgdvDQ$MNVY.9ajgdvDQ')]
    known_alternate_hashes = [('crypt$$c2M87q...WWcU', 'password', 'crypt$c2$c2M87q...WWcU')]
    known_unidentified_hashes = ['sha1$aa$bb']
    known_malformed_hashes = ['crypt$c2$c2M87q', 'crypt$f$c2M87q...WWcU', 'crypt$ffe86$c2M87q...WWcU']