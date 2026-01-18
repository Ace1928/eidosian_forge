from __future__ import with_statement
from binascii import unhexlify
import contextlib
from functools import wraps, partial
import hashlib
import logging; log = logging.getLogger(__name__)
import random
import re
import os
import sys
import tempfile
import threading
import time
from passlib.exc import PasslibHashWarning, PasslibConfigWarning
from passlib.utils.compat import PY3, JYTHON
import warnings
from warnings import warn
from passlib import exc
from passlib.exc import MissingBackendError
import passlib.registry as registry
from passlib.tests.backports import TestCase as _TestCase, skip, skipIf, skipUnless, SkipTest
from passlib.utils import has_rounds_info, has_salt_info, rounds_cost_values, \
from passlib.utils.compat import iteritems, irange, u, unicode, PY2, nullcontext
from passlib.utils.decor import classproperty
import passlib.utils.handlers as uh
class UserHandlerMixin(HandlerCase):
    """helper for handlers w/ 'user' context kwd; mixin for HandlerCase

    this overrides the HandlerCase test harness methods
    so that a username is automatically inserted to hash/verify
    calls. as well, passing in a pair of strings as the password
    will be interpreted as (secret,user)
    """
    default_user = 'user'
    requires_user = True
    user_case_insensitive = False
    __unittest_skip = True

    def test_80_user(self):
        """test user context keyword"""
        handler = self.handler
        password = 'stub'
        hash = handler.hash(password, user=self.default_user)
        if self.requires_user:
            self.assertRaises(TypeError, handler.hash, password)
            self.assertRaises(TypeError, handler.genhash, password, hash)
            self.assertRaises(TypeError, handler.verify, password, hash)
        else:
            handler.hash(password)
            handler.genhash(password, hash)
            handler.verify(password, hash)

    def test_81_user_case(self):
        """test user case sensitivity"""
        lower = self.default_user.lower()
        upper = lower.upper()
        hash = self.do_encrypt('stub', context=dict(user=lower))
        if self.user_case_insensitive:
            self.assertTrue(self.do_verify('stub', hash, user=upper), 'user should not be case sensitive')
        else:
            self.assertFalse(self.do_verify('stub', hash, user=upper), 'user should be case sensitive')

    def test_82_user_salt(self):
        """test user used as salt"""
        config = self.do_stub_encrypt()
        h1 = self.do_genhash('stub', config, user='admin')
        h2 = self.do_genhash('stub', config, user='admin')
        self.assertEqual(h2, h1)
        h3 = self.do_genhash('stub', config, user='root')
        self.assertNotEqual(h3, h1)

    def populate_context(self, secret, kwds):
        """insert username into kwds"""
        if isinstance(secret, tuple):
            secret, user = secret
        elif not self.requires_user:
            return secret
        else:
            user = self.default_user
        if 'user' not in kwds:
            kwds['user'] = user
        return secret

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):
        context_map = HandlerCase.FuzzHashGenerator.context_map.copy()
        context_map.update(user='random_user')
        user_alphabet = u('asdQWE123')

        def random_user(self):
            rng = self.rng
            if not self.test.requires_user and rng.random() < 0.1:
                return None
            return getrandstr(rng, self.user_alphabet, rng.randint(2, 10))