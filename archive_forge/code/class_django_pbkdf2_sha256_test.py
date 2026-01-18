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
class django_pbkdf2_sha256_test(HandlerCase, _DjangoHelper):
    """test django_pbkdf2_sha256"""
    handler = hash.django_pbkdf2_sha256
    known_correct_hashes = [('not a password', 'pbkdf2_sha256$10000$kjVJaVz6qsnJ$5yPHw3rwJGECpUf70daLGhOrQ5+AMxIJdz1c3bqK1Rs='), (UPASS_TABLE, 'pbkdf2_sha256$10000$bEwAfNrH1TlQ$OgYUblFNUX1B8GfMqaCYUK/iHyO0pa7STTDdaEJBuY0=')]