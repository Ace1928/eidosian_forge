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
class _DjangoHelper(TestCase):
    """
    mixin for HandlerCase subclasses that are testing a hasher
    which is also present in django.
    """
    __unittest_skip = True
    min_django_version = MIN_DJANGO_VERSION
    max_django_version = None

    def _require_django_support(self):
        if DJANGO_VERSION < self.min_django_version:
            raise self.skipTest('Django >= %s not installed' % vstr(self.min_django_version))
        if self.max_django_version and DJANGO_VERSION > self.max_django_version:
            raise self.skipTest('Django <= %s not installed' % vstr(self.max_django_version))
        name = self.handler.django_name
        if not check_django_hasher_has_backend(name):
            raise self.skipTest('django hasher %r not available' % name)
        return True
    extra_fuzz_verifiers = HandlerCase.fuzz_verifiers + ('fuzz_verifier_django',)

    def fuzz_verifier_django(self):
        try:
            self._require_django_support()
        except SkipTest:
            return None
        from django.contrib.auth.hashers import check_password

        def verify_django(secret, hash):
            """django/check_password"""
            if self.handler.name == 'django_bcrypt' and hash.startswith('bcrypt$$2y$'):
                hash = hash.replace('$$2y$', '$$2a$')
            if isinstance(secret, bytes):
                secret = secret.decode('utf-8')
            return check_password(secret, hash)
        return verify_django

    def test_90_django_reference(self):
        """run known correct hashes through Django's check_password()"""
        self._require_django_support()
        from django.contrib.auth.hashers import check_password
        assert self.known_correct_hashes
        for secret, hash in self.iter_known_hashes():
            self.assertTrue(check_password(secret, hash), 'secret=%r hash=%r failed to verify' % (secret, hash))
            self.assertFalse(check_password('x' + secret, hash), 'mangled secret=%r hash=%r incorrect verified' % (secret, hash))

    def test_91_django_generation(self):
        """test against output of Django's make_password()"""
        self._require_django_support()
        from passlib.utils import tick
        from django.contrib.auth.hashers import make_password
        name = self.handler.django_name
        end = tick() + self.max_fuzz_time / 2
        generator = self.FuzzHashGenerator(self, self.getRandom())
        while tick() < end:
            secret, other = generator.random_password_pair()
            if not secret:
                continue
            if isinstance(secret, bytes):
                secret = secret.decode('utf-8')
            hash = make_password(secret, hasher=name)
            self.assertTrue(self.do_identify(hash))
            self.assertTrue(self.do_verify(secret, hash))
            self.assertFalse(self.do_verify(other, hash))