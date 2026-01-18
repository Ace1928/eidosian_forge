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
class django_salted_md5_test(HandlerCase, _DjangoHelper):
    """test django_salted_md5"""
    handler = hash.django_salted_md5
    max_django_version = (1, 9)
    known_correct_hashes = [('password', 'md5$123abcdef$c8272612932975ee80e8a35995708e80'), ('test', 'md5$3OpqnFAHW5CT$54b29300675271049a1ebae07b395e20'), (UPASS_USD, 'md5$c2e86$92105508419a81a6babfaecf876a2fa0'), (UPASS_TABLE, 'md5$d9eb8$01495b32852bffb27cf5d4394fe7a54c')]
    known_unidentified_hashes = ['sha1$aa$bb']
    known_malformed_hashes = ['md5$aa$bb']

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):

        def random_salt_size(self):
            handler = self.handler
            default = handler.default_salt_size
            assert handler.min_salt_size == 0
            lower = 1
            upper = handler.max_salt_size or default * 4
            return self.randintgauss(lower, upper, default, default * 0.5)