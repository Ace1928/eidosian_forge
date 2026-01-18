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
class _ExtensionSupport(object):
    """
    test support funcs for loading/unloading extension.
    this class is mixed in to various TestCase subclasses.
    """

    @classmethod
    def _iter_patch_candidates(cls):
        """helper to scan for monkeypatches.

        returns tuple containing:
        * object (module or class)
        * attribute of object
        * value of attribute
        * whether it should or should not be patched
        """
        from django.contrib.auth import models, hashers
        user_attrs = ['check_password', 'set_password']
        model_attrs = ['check_password', 'make_password']
        hasher_attrs = ['check_password', 'make_password', 'get_hasher', 'identify_hasher', 'get_hashers']
        objs = [(models, model_attrs), (models.User, user_attrs), (hashers, hasher_attrs)]
        for obj, patched in objs:
            for attr in dir(obj):
                if attr.startswith('_'):
                    continue
                value = obj.__dict__.get(attr, UNSET)
                if value is UNSET and attr not in patched:
                    continue
                value = get_method_function(value)
                source = getattr(value, '__module__', None)
                if source:
                    yield (obj, attr, source, attr in patched)

    def assert_unpatched(self):
        """
        test that django is in unpatched state
        """
        mod = sys.modules.get('passlib.ext.django.models')
        self.assertFalse(mod and mod.adapter.patched, 'patch should not be enabled')
        for obj, attr, source, patched in self._iter_patch_candidates():
            if patched:
                self.assertTrue(source.startswith('django.contrib.auth.'), 'obj=%r attr=%r was not reverted: %r' % (obj, attr, source))
            else:
                self.assertFalse(source.startswith('passlib.'), 'obj=%r attr=%r should not have been patched: %r' % (obj, attr, source))

    def assert_patched(self, context=None):
        """
        helper to ensure django HAS been patched, and is using specified config
        """
        mod = sys.modules.get('passlib.ext.django.models')
        self.assertTrue(mod and mod.adapter.patched, 'patch should have been enabled')
        for obj, attr, source, patched in self._iter_patch_candidates():
            if patched:
                self.assertTrue(source == 'passlib.ext.django.utils', 'obj=%r attr=%r should have been patched: %r' % (obj, attr, source))
            else:
                self.assertFalse(source.startswith('passlib.'), 'obj=%r attr=%r should not have been patched: %r' % (obj, attr, source))
        if context is not None:
            context = CryptContext._norm_source(context)
            self.assertEqual(mod.password_context.to_dict(resolve=True), context.to_dict(resolve=True))
    _config_keys = ['PASSLIB_CONFIG', 'PASSLIB_CONTEXT', 'PASSLIB_GET_CATEGORY']

    def load_extension(self, check=True, **kwds):
        """
        helper to load extension with specified config & patch django
        """
        self.unload_extension()
        if check:
            config = kwds.get('PASSLIB_CONFIG') or kwds.get('PASSLIB_CONTEXT')
        for key in self._config_keys:
            kwds.setdefault(key, UNSET)
        update_settings(**kwds)
        import passlib.ext.django.models
        if check:
            self.assert_patched(context=config)

    def unload_extension(self):
        """
        helper to remove patches and unload extension
        """
        mod = sys.modules.get('passlib.ext.django.models')
        if mod:
            mod.adapter.remove_patch()
            del sys.modules['passlib.ext.django.models']
        update_settings(**dict(((key, UNSET) for key in self._config_keys)))
        self.assert_unpatched()