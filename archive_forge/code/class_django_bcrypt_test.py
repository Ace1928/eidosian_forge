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
@skipUnless(hash.bcrypt.has_backend(), 'no bcrypt backends available')
class django_bcrypt_test(HandlerCase, _DjangoHelper):
    """test django_bcrypt"""
    handler = hash.django_bcrypt
    max_django_version = (2, 0)
    fuzz_salts_need_bcrypt_repair = True
    known_correct_hashes = [('', 'bcrypt$$2a$06$DCq7YPn5Rq63x1Lad4cll.TV4S6ytwfsfvkgY8jIucDrjc8deX1s.'), ('abcdefghijklmnopqrstuvwxyz', 'bcrypt$$2a$10$fVH8e28OQRj9tqiDXs1e1uxpsjN0c7II7YPKXua2NAKYvM6iQk7dq'), (UPASS_TABLE, 'bcrypt$$2a$05$Z17AXnnlpzddNUvnC6cZNOSwMA/8oNiKnHTHTwLlBijfucQQlHjaG')]

    def populate_settings(self, kwds):
        kwds.setdefault('rounds', 4)
        super(django_bcrypt_test, self).populate_settings(kwds)

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):

        def random_rounds(self):
            return self.randintgauss(5, 8, 6, 1)

        def random_ident(self):
            return None