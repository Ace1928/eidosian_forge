from __future__ import absolute_import, division, print_function
import logging; log = logging.getLogger(__name__)
import sys
import re
from passlib import apps as _apps, exc, registry
from passlib.apps import django10_context, django14_context, django16_context
from passlib.context import CryptContext
from passlib.ext.django.utils import (
from passlib.utils.compat import iteritems, get_method_function, u
from passlib.utils.decor import memoized_property
from passlib.tests.utils import TestCase, TEST_MODE, handler_derived_from
from passlib.tests.test_handlers import get_handler_case
from passlib.hash import django_pbkdf2_sha256
class DjangoExtensionTest(_ExtensionTest):
    """
    test the ``passlib.ext.django`` plugin
    """
    descriptionPrefix = 'passlib.ext.django plugin'

    def test_00_patch_control(self):
        """test set_django_password_context patch/unpatch"""
        self.load_extension(PASSLIB_CONFIG='disabled', check=False)
        self.assert_unpatched()
        with self.assertWarningList('PASSLIB_CONFIG=None is deprecated'):
            self.load_extension(PASSLIB_CONFIG=None, check=False)
        self.assert_unpatched()
        self.load_extension(PASSLIB_CONFIG='django-1.0', check=False)
        self.assert_patched(context=django10_context)
        self.unload_extension()
        self.load_extension(PASSLIB_CONFIG='django-1.4', check=False)
        self.assert_patched(context=django14_context)
        self.unload_extension()

    def test_01_overwrite_detection(self):
        """test detection of foreign monkeypatching"""
        config = '[passlib]\nschemes=des_crypt\n'
        self.load_extension(PASSLIB_CONFIG=config)
        import django.contrib.auth.models as models
        from passlib.ext.django.models import adapter

        def dummy():
            pass
        orig = models.User.set_password
        models.User.set_password = dummy
        with self.assertWarningList('another library has patched.*User\\.set_password'):
            adapter._manager.check_all()
        models.User.set_password = orig
        orig = models.check_password
        models.check_password = dummy
        with self.assertWarningList('another library has patched.*models:check_password'):
            adapter._manager.check_all()
        models.check_password = orig

    def test_02_handler_wrapper(self):
        """test Hasher-compatible handler wrappers"""
        from django.contrib.auth import hashers
        passlib_to_django = DjangoTranslator().passlib_to_django
        if DJANGO_VERSION > (1, 10):
            self.assertRaises(ValueError, passlib_to_django, 'hex_md5')
        else:
            hasher = passlib_to_django('hex_md5')
            self.assertIsInstance(hasher, hashers.UnsaltedMD5PasswordHasher)
        hasher = passlib_to_django('django_bcrypt')
        self.assertIsInstance(hasher, hashers.BCryptPasswordHasher)
        from passlib.hash import sha256_crypt
        hasher = passlib_to_django('sha256_crypt')
        self.assertEqual(hasher.algorithm, 'passlib_sha256_crypt')
        encoded = hasher.encode('stub')
        self.assertTrue(sha256_crypt.verify('stub', encoded))
        self.assertTrue(hasher.verify('stub', encoded))
        self.assertFalse(hasher.verify('xxxx', encoded))
        encoded = hasher.encode('stub', 'abcd' * 4, rounds=1234)
        self.assertEqual(encoded, '$5$rounds=1234$abcdabcdabcdabcd$v2RWkZQzctPdejyRqmmTDQpZN6wTh7.RUy9zF2LftT6')
        self.assertEqual(hasher.safe_summary(encoded), {'algorithm': 'sha256_crypt', 'salt': u('abcdab**********'), 'rounds': 1234, 'hash': u('v2RWkZ*************************************')})
        self.assertRaises(KeyError, passlib_to_django, 'does_not_exist')

    def test_11_config_disabled(self):
        """test PASSLIB_CONFIG='disabled'"""
        with self.assertWarningList('PASSLIB_CONFIG=None is deprecated'):
            self.load_extension(PASSLIB_CONFIG=None, check=False)
        self.assert_unpatched()
        self.load_extension(PASSLIB_CONFIG='disabled', check=False)
        self.assert_unpatched()

    def test_12_config_presets(self):
        """test PASSLIB_CONFIG='<preset>'"""
        self.load_extension(PASSLIB_CONTEXT='django-default', check=False)
        ctx = django16_context
        self.assert_patched(ctx)
        self.load_extension(PASSLIB_CONFIG='django-1.0', check=False)
        self.assert_patched(django10_context)
        self.load_extension(PASSLIB_CONFIG='django-1.4', check=False)
        self.assert_patched(django14_context)

    def test_13_config_defaults(self):
        """test PASSLIB_CONFIG default behavior"""
        from passlib.ext.django.utils import PASSLIB_DEFAULT
        default = CryptContext.from_string(PASSLIB_DEFAULT)
        self.load_extension()
        self.assert_patched(PASSLIB_DEFAULT)
        self.load_extension(PASSLIB_CONTEXT='passlib-default', check=False)
        self.assert_patched(PASSLIB_DEFAULT)
        self.load_extension(PASSLIB_CONTEXT=PASSLIB_DEFAULT, check=False)
        self.assert_patched(PASSLIB_DEFAULT)

    def test_14_config_invalid(self):
        """test PASSLIB_CONFIG type checks"""
        update_settings(PASSLIB_CONTEXT=123, PASSLIB_CONFIG=UNSET)
        self.assertRaises(TypeError, __import__, 'passlib.ext.django.models')
        self.unload_extension()
        update_settings(PASSLIB_CONFIG='missing-preset', PASSLIB_CONTEXT=UNSET)
        self.assertRaises(ValueError, __import__, 'passlib.ext.django.models')

    def test_21_category_setting(self):
        """test PASSLIB_GET_CATEGORY parameter"""
        config = dict(schemes=['sha256_crypt'], sha256_crypt__default_rounds=1000, staff__sha256_crypt__default_rounds=2000, superuser__sha256_crypt__default_rounds=3000)
        from passlib.hash import sha256_crypt

        def run(**kwds):
            """helper to take in user opts, return rounds used in password"""
            user = FakeUser(**kwds)
            user.set_password('stub')
            return sha256_crypt.from_string(user.password).rounds
        self.load_extension(PASSLIB_CONFIG=config)
        self.assertEqual(run(), 1000)
        self.assertEqual(run(is_staff=True), 2000)
        self.assertEqual(run(is_superuser=True), 3000)

        def get_category(user):
            return user.first_name or None
        self.load_extension(PASSLIB_CONTEXT=config, PASSLIB_GET_CATEGORY=get_category)
        self.assertEqual(run(), 1000)
        self.assertEqual(run(first_name='other'), 1000)
        self.assertEqual(run(first_name='staff'), 2000)
        self.assertEqual(run(first_name='superuser'), 3000)

        def get_category(user):
            return None
        self.load_extension(PASSLIB_CONTEXT=config, PASSLIB_GET_CATEGORY=get_category)
        self.assertEqual(run(), 1000)
        self.assertEqual(run(first_name='other'), 1000)
        self.assertEqual(run(first_name='staff', is_staff=True), 1000)
        self.assertEqual(run(first_name='superuser', is_superuser=True), 1000)
        self.assertRaises(TypeError, self.load_extension, PASSLIB_CONTEXT=config, PASSLIB_GET_CATEGORY='x')