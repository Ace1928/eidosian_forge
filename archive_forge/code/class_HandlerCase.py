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
class HandlerCase(TestCase):
    """base class for testing password hash handlers (esp passlib.utils.handlers subclasses)

    In order to use this to test a handler,
    create a subclass will all the appropriate attributes
    filled as listed in the example below,
    and run the subclass via unittest.

    .. todo::

        Document all of the options HandlerCase offers.

    .. note::

        This is subclass of :class:`unittest.TestCase`
        (or :class:`unittest2.TestCase` if available).
    """
    handler = None
    backend = None
    known_correct_hashes = []
    known_correct_configs = []
    known_alternate_hashes = []
    known_unidentified_hashes = []
    known_malformed_hashes = []
    known_other_hashes = [('des_crypt', '6f8c114b58f2c'), ('md5_crypt', '$1$dOHYPKoP$tnxS1T8Q6VVn3kpV8cN6o.'), ('sha512_crypt', '$6$rounds=123456$asaltof16chars..$BtCwjqMJGx5hrJhZywWvt0RLE8uZ4oPwcelCjmw2kSYu.Ec6ycULevoBK25fs2xXgMNrCzIMVcgEJAstJeonj1')]
    stock_passwords = [u('test'), u('€¥$'), b'\xe2\x82\xac\xc2\xa5$']
    secret_case_insensitive = False
    accepts_all_hashes = False
    disabled_contains_salt = False
    filter_config_warnings = False

    @classproperty
    def forbidden_characters(cls):
        if 'os_crypt' in getattr(cls.handler, 'backends', ()):
            return b'\x00'
        return None
    __unittest_skip = True

    @property
    def descriptionPrefix(self):
        handler = self.handler
        name = handler.name
        if hasattr(handler, 'get_backend'):
            name += ' (%s backend)' % (handler.get_backend(),)
        return name

    @classmethod
    def iter_known_hashes(cls):
        """iterate through known (secret, hash) pairs"""
        for secret, hash in cls.known_correct_hashes:
            yield (secret, hash)
        for config, secret, hash in cls.known_correct_configs:
            yield (secret, hash)
        for alt, secret, hash in cls.known_alternate_hashes:
            yield (secret, hash)

    def get_sample_hash(self):
        """test random sample secret/hash pair"""
        known = list(self.iter_known_hashes())
        return self.getRandom().choice(known)

    def check_verify(self, secret, hash, msg=None, negate=False):
        """helper to check verify() outcome, honoring is_disabled_handler"""
        result = self.do_verify(secret, hash)
        self.assertTrue(result is True or result is False, 'verify() returned non-boolean value: %r' % (result,))
        if self.handler.is_disabled or negate:
            if not result:
                return
            if not msg:
                msg = 'verify incorrectly returned True: secret=%r, hash=%r' % (secret, hash)
            raise self.failureException(msg)
        else:
            if result:
                return
            if not msg:
                msg = 'verify failed: secret=%r, hash=%r' % (secret, hash)
            raise self.failureException(msg)

    def check_returned_native_str(self, result, func_name):
        self.assertIsInstance(result, str, '%s() failed to return native string: %r' % (func_name, result))

    def populate_settings(self, kwds):
        """subclassable method to populate default settings"""
        handler = self.handler
        if 'rounds' in handler.setting_kwds and 'rounds' not in kwds:
            mn = handler.min_rounds
            df = handler.default_rounds
            if TEST_MODE(max='quick'):
                kwds['rounds'] = max(3, mn)
            else:
                factor = 3
                if getattr(handler, 'rounds_cost', None) == 'log2':
                    df -= factor
                else:
                    df //= 1 << factor
                kwds['rounds'] = max(3, mn, df)

    def populate_context(self, secret, kwds):
        """subclassable method allowing 'secret' to be encode context kwds"""
        return secret

    def do_encrypt(self, secret, use_encrypt=False, handler=None, context=None, **settings):
        """call handler's hash() method with specified options"""
        self.populate_settings(settings)
        if context is None:
            context = {}
        secret = self.populate_context(secret, context)
        if use_encrypt:
            warnings = []
            if settings:
                context.update(**settings)
                warnings.append('passing settings to.*is deprecated')
            with self.assertWarningList(warnings):
                return (handler or self.handler).encrypt(secret, **context)
        else:
            return (handler or self.handler).using(**settings).hash(secret, **context)

    def do_verify(self, secret, hash, handler=None, **kwds):
        """call handler's verify method"""
        secret = self.populate_context(secret, kwds)
        return (handler or self.handler).verify(secret, hash, **kwds)

    def do_identify(self, hash):
        """call handler's identify method"""
        return self.handler.identify(hash)

    def do_genconfig(self, **kwds):
        """call handler's genconfig method with specified options"""
        self.populate_settings(kwds)
        return self.handler.genconfig(**kwds)

    def do_genhash(self, secret, config, **kwds):
        """call handler's genhash method with specified options"""
        secret = self.populate_context(secret, kwds)
        return self.handler.genhash(secret, config, **kwds)

    def do_stub_encrypt(self, handler=None, context=None, **settings):
        """
        return sample hash for handler, w/o caring if digest is valid
        (uses some monkeypatching to minimize digest calculation cost)
        """
        handler = (handler or self.handler).using(**settings)
        if context is None:
            context = {}
        secret = self.populate_context('', context)
        with patch_calc_min_rounds(handler):
            return handler.hash(secret, **context)
    _BACKEND_NOT_AVAILABLE = 'backend not available'

    @classmethod
    def _get_skip_backend_reason(cls, backend):
        """
        helper for create_backend_case() --
        returns reason to skip backend, or None if backend should be tested
        """
        handler = cls.handler
        if not is_default_backend(handler, backend) and (not TEST_MODE('full')):
            return 'only default backend is being tested'
        if handler.has_backend(backend):
            return None
        return cls._BACKEND_NOT_AVAILABLE

    @classmethod
    def create_backend_case(cls, backend):
        handler = cls.handler
        name = handler.name
        assert hasattr(handler, 'backends'), 'handler must support uh.HasManyBackends protocol'
        assert backend in handler.backends, 'unknown backend: %r' % (backend,)
        bases = (cls,)
        if backend == 'os_crypt':
            bases += (OsCryptMixin,)
        subcls = type('%s_%s_test' % (name, backend), bases, dict(descriptionPrefix='%s (%s backend)' % (name, backend), backend=backend, _skip_backend_reason=cls._get_skip_backend_reason(backend), __module__=cls.__module__))
        return subcls
    _skip_backend_reason = None

    def _test_requires_backend(self):
        """
        check if current test method decorated with doesnt_require_backend() helper
        """
        meth = getattr(self, self._testMethodName, None)
        return not getattr(meth, '_doesnt_require_backend', False)

    def setUp(self):
        test_requires_backend = self._test_requires_backend()
        if test_requires_backend and self._skip_backend_reason:
            raise self.skipTest(self._skip_backend_reason)
        super(HandlerCase, self).setUp()
        handler = self.handler
        backend = self.backend
        if backend:
            if not hasattr(handler, 'set_backend'):
                raise RuntimeError("handler doesn't support multiple backends")
            try:
                self.addCleanup(handler.set_backend, handler.get_backend())
                handler.set_backend(backend)
            except uh.exc.MissingBackendError:
                if test_requires_backend:
                    raise
        from passlib.utils import handlers
        self.patchAttr(handlers, 'rng', self.getRandom('salt generator'))

    def test_01_required_attributes(self):
        """validate required attributes"""
        handler = self.handler

        def ga(name):
            return getattr(handler, name, None)
        name = ga('name')
        self.assertTrue(name, 'name not defined:')
        self.assertIsInstance(name, str, 'name must be native str')
        self.assertTrue(name.lower() == name, 'name not lower-case:')
        self.assertTrue(re.match('^[a-z0-9_]+$', name), 'name must be alphanum + underscore: %r' % (name,))
        settings = ga('setting_kwds')
        self.assertTrue(settings is not None, 'setting_kwds must be defined:')
        self.assertIsInstance(settings, tuple, 'setting_kwds must be a tuple:')
        context = ga('context_kwds')
        self.assertTrue(context is not None, 'context_kwds must be defined:')
        self.assertIsInstance(context, tuple, 'context_kwds must be a tuple:')

    def test_02_config_workflow(self):
        """test basic config-string workflow

        this tests that genconfig() returns the expected types,
        and that identify() and genhash() handle the result correctly.
        """
        config = self.do_genconfig()
        self.check_returned_native_str(config, 'genconfig')
        result = self.do_genhash('stub', config)
        self.check_returned_native_str(result, 'genhash')
        self.do_verify('', config)
        self.assertTrue(self.do_identify(config), 'identify() failed to identify genconfig() output: %r' % (config,))

    def test_02_using_workflow(self):
        """test basic using() workflow"""
        handler = self.handler
        subcls = handler.using()
        self.assertIsNot(subcls, handler)
        self.assertEqual(subcls.name, handler.name)

    def test_03_hash_workflow(self, use_16_legacy=False):
        """test basic hash-string workflow.

        this tests that hash()'s hashes are accepted
        by verify() and identify(), and regenerated correctly by genhash().
        the test is run against a couple of different stock passwords.
        """
        wrong_secret = 'stub'
        for secret in self.stock_passwords:
            result = self.do_encrypt(secret, use_encrypt=use_16_legacy)
            self.check_returned_native_str(result, 'hash')
            self.check_verify(secret, result)
            self.check_verify(wrong_secret, result, negate=True)
            other = self.do_genhash(secret, result)
            self.check_returned_native_str(other, 'genhash')
            if self.handler.is_disabled and self.disabled_contains_salt:
                self.assertNotEqual(other, result, 'genhash() failed to salt result hash: secret=%r hash=%r: result=%r' % (secret, result, other))
            else:
                self.assertEqual(other, result, 'genhash() failed to reproduce hash: secret=%r hash=%r: result=%r' % (secret, result, other))
            other = self.do_genhash(wrong_secret, result)
            self.check_returned_native_str(other, 'genhash')
            if self.handler.is_disabled and (not self.disabled_contains_salt):
                self.assertEqual(other, result, 'genhash() failed to reproduce disabled-hash: secret=%r hash=%r other_secret=%r: result=%r' % (secret, result, wrong_secret, other))
            else:
                self.assertNotEqual(other, result, 'genhash() duplicated hash: secret=%r hash=%r wrong_secret=%r: result=%r' % (secret, result, wrong_secret, other))
            self.assertTrue(self.do_identify(result))

    def test_03_legacy_hash_workflow(self):
        """test hash-string workflow with legacy .encrypt() & .genhash() methods"""
        self.test_03_hash_workflow(use_16_legacy=True)

    def test_04_hash_types(self):
        """test hashes can be unicode or bytes"""
        result = self.do_encrypt(tonn('stub'))
        self.check_returned_native_str(result, 'hash')
        self.check_verify('stub', tonn(result))
        self.check_verify(tonn('stub'), tonn(result))
        other = self.do_genhash('stub', tonn(result))
        self.check_returned_native_str(other, 'genhash')
        if self.handler.is_disabled and self.disabled_contains_salt:
            self.assertNotEqual(other, result)
        else:
            self.assertEqual(other, result)
        other = self.do_genhash(tonn('stub'), tonn(result))
        self.check_returned_native_str(other, 'genhash')
        if self.handler.is_disabled and self.disabled_contains_salt:
            self.assertNotEqual(other, result)
        else:
            self.assertEqual(other, result)
        self.assertTrue(self.do_identify(tonn(result)))

    def test_05_backends(self):
        """test multi-backend support"""
        handler = self.handler
        if not hasattr(handler, 'set_backend'):
            raise self.skipTest('handler only has one backend')
        self.addCleanup(handler.set_backend, handler.get_backend())
        for backend in handler.backends:
            self.assertIsInstance(backend, str)
            self.assertNotIn(backend, RESERVED_BACKEND_NAMES, 'invalid backend name: %r' % (backend,))
            ret = handler.has_backend(backend)
            if ret is True:
                handler.set_backend(backend)
                self.assertEqual(handler.get_backend(), backend)
            elif ret is False:
                self.assertRaises(MissingBackendError, handler.set_backend, backend)
            else:
                raise TypeError('has_backend(%r) returned invalid value: %r' % (backend, ret))

    def require_salt(self):
        if 'salt' not in self.handler.setting_kwds:
            raise self.skipTest("handler doesn't have salt")

    def require_salt_info(self):
        self.require_salt()
        if not has_salt_info(self.handler):
            raise self.skipTest("handler doesn't provide salt info")

    def test_10_optional_salt_attributes(self):
        """validate optional salt attributes"""
        self.require_salt_info()
        AssertionError = self.failureException
        cls = self.handler
        mx_set = cls.max_salt_size is not None
        if mx_set and cls.max_salt_size < 1:
            raise AssertionError('max_salt_chars must be >= 1')
        if cls.min_salt_size < 0:
            raise AssertionError('min_salt_chars must be >= 0')
        if mx_set and cls.min_salt_size > cls.max_salt_size:
            raise AssertionError('min_salt_chars must be <= max_salt_chars')
        if cls.default_salt_size < cls.min_salt_size:
            raise AssertionError('default_salt_size must be >= min_salt_size')
        if mx_set and cls.default_salt_size > cls.max_salt_size:
            raise AssertionError('default_salt_size must be <= max_salt_size')
        if 'salt_size' not in cls.setting_kwds and (not mx_set or cls.default_salt_size < cls.max_salt_size):
            warn("%s: hash handler supports range of salt sizes, but doesn't offer 'salt_size' setting" % (cls.name,))
        if cls.salt_chars:
            if not cls.default_salt_chars:
                raise AssertionError('default_salt_chars must not be empty')
            for c in cls.default_salt_chars:
                if c not in cls.salt_chars:
                    raise AssertionError('default_salt_chars must be subset of salt_chars: %r not in salt_chars' % (c,))
        elif not cls.default_salt_chars:
            raise AssertionError('default_salt_chars MUST be specified if salt_chars is empty')

    @property
    def salt_bits(self):
        """calculate number of salt bits in hash"""
        handler = self.handler
        assert has_salt_info(handler), 'need explicit bit-size for ' + handler.name
        from math import log
        return int(handler.default_salt_size * log(len(handler.default_salt_chars), 2))

    def test_11_unique_salt(self):
        """test hash() / genconfig() creates new salt each time"""
        self.require_salt()
        samples = max(1, 4 + 12 - self.salt_bits)

        def sampler(func):
            value1 = func()
            for _ in irange(samples):
                value2 = func()
                if value1 != value2:
                    return
            raise self.failureException('failed to find different salt after %d samples' % (samples,))
        sampler(self.do_genconfig)
        sampler(lambda: self.do_encrypt('stub'))

    def test_12_min_salt_size(self):
        """test hash() / genconfig() honors min_salt_size"""
        self.require_salt_info()
        handler = self.handler
        salt_char = handler.salt_chars[0:1]
        min_size = handler.min_salt_size
        s1 = salt_char * min_size
        self.do_genconfig(salt=s1)
        self.do_encrypt('stub', salt_size=min_size)
        if min_size > 0:
            self.assertRaises(ValueError, self.do_genconfig, salt=s1[:-1])
        self.assertRaises(ValueError, self.do_encrypt, 'stub', salt_size=min_size - 1)

    def test_13_max_salt_size(self):
        """test hash() / genconfig() honors max_salt_size"""
        self.require_salt_info()
        handler = self.handler
        max_size = handler.max_salt_size
        salt_char = handler.salt_chars[0:1]
        if max_size is None or max_size > 1 << 20:
            s1 = salt_char * 1024
            c1 = self.do_stub_encrypt(salt=s1)
            c2 = self.do_stub_encrypt(salt=s1 + salt_char)
            self.assertNotEqual(c1, c2)
            self.do_stub_encrypt(salt_size=1024)
        else:
            s1 = salt_char * max_size
            c1 = self.do_stub_encrypt(salt=s1)
            self.do_stub_encrypt(salt_size=max_size)
            s2 = s1 + salt_char
            self.assertRaises(ValueError, self.do_stub_encrypt, salt=s2)
            self.assertRaises(ValueError, self.do_stub_encrypt, salt_size=max_size + 1)
            if has_relaxed_setting(handler):
                with warnings.catch_warnings(record=True):
                    c2 = self.do_stub_encrypt(salt=s2, relaxed=True)
                self.assertEqual(c2, c1)
            if handler.min_salt_size < max_size:
                c3 = self.do_stub_encrypt(salt=s1[:-1])
                self.assertNotEqual(c3, c1)
    fuzz_salts_need_bcrypt_repair = False

    def prepare_salt(self, salt):
        """prepare generated salt"""
        if self.fuzz_salts_need_bcrypt_repair:
            from passlib.utils.binary import bcrypt64
            salt = bcrypt64.repair_unused(salt)
        return salt

    def test_14_salt_chars(self):
        """test hash() honors salt_chars"""
        self.require_salt_info()
        handler = self.handler
        mx = handler.max_salt_size
        mn = handler.min_salt_size
        cs = handler.salt_chars
        raw = isinstance(cs, bytes)
        for salt in batch(cs, mx or 32):
            if len(salt) < mn:
                salt = repeat_string(salt, mn)
            salt = self.prepare_salt(salt)
            self.do_stub_encrypt(salt=salt)
        source = u('\x00ÿ')
        if raw:
            source = source.encode('latin-1')
        chunk = max(mn, 1)
        for c in source:
            if c not in cs:
                self.assertRaises(ValueError, self.do_stub_encrypt, salt=c * chunk, __msg__='invalid salt char %r:' % (c,))

    @property
    def salt_type(self):
        """hack to determine salt keyword's datatype"""
        if getattr(self.handler, '_salt_is_bytes', False):
            return bytes
        else:
            return unicode

    def test_15_salt_type(self):
        """test non-string salt values"""
        self.require_salt()
        salt_type = self.salt_type
        salt_size = getattr(self.handler, 'min_salt_size', 0) or 8

        class fake(object):
            pass
        self.assertRaises(TypeError, self.do_encrypt, 'stub', salt=fake())
        if salt_type is not unicode:
            self.assertRaises(TypeError, self.do_encrypt, 'stub', salt=u('x') * salt_size)
        if not (salt_type is bytes or (PY2 and salt_type is unicode)):
            self.assertRaises(TypeError, self.do_encrypt, 'stub', salt=b'x' * salt_size)

    def test_using_salt_size(self):
        """Handler.using() -- default_salt_size"""
        self.require_salt_info()
        handler = self.handler
        mn = handler.min_salt_size
        mx = handler.max_salt_size
        df = handler.default_salt_size
        self.assertRaises(ValueError, handler.using, default_salt_size=-1)
        with self.assertWarningList([PasslibHashWarning]):
            temp = handler.using(default_salt_size=-1, relaxed=True)
        self.assertEqual(temp.default_salt_size, mn)
        if mx:
            self.assertRaises(ValueError, handler.using, default_salt_size=mx + 1)
            with self.assertWarningList([PasslibHashWarning]):
                temp = handler.using(default_salt_size=mx + 1, relaxed=True)
            self.assertEqual(temp.default_salt_size, mx)
        if mn != mx:
            temp = handler.using(default_salt_size=mn + 1)
            self.assertEqual(temp.default_salt_size, mn + 1)
            self.assertEqual(handler.default_salt_size, df)
            temp = handler.using(default_salt_size=mn + 2)
            self.assertEqual(temp.default_salt_size, mn + 2)
            self.assertEqual(handler.default_salt_size, df)
        if mn == mx:
            ref = mn
        else:
            ref = mn + 1
        temp = handler.using(default_salt_size=str(ref))
        self.assertEqual(temp.default_salt_size, ref)
        self.assertRaises(ValueError, handler.using, default_salt_size=str(ref) + 'xxx')
        temp = handler.using(salt_size=ref)
        self.assertEqual(temp.default_salt_size, ref)

    def require_rounds_info(self):
        if not has_rounds_info(self.handler):
            raise self.skipTest('handler lacks rounds attributes')

    def test_20_optional_rounds_attributes(self):
        """validate optional rounds attributes"""
        self.require_rounds_info()
        cls = self.handler
        AssertionError = self.failureException
        if cls.max_rounds is None:
            raise AssertionError('max_rounds not specified')
        if cls.max_rounds < 1:
            raise AssertionError('max_rounds must be >= 1')
        if cls.min_rounds < 0:
            raise AssertionError('min_rounds must be >= 0')
        if cls.min_rounds > cls.max_rounds:
            raise AssertionError('min_rounds must be <= max_rounds')
        if cls.default_rounds is not None:
            if cls.default_rounds < cls.min_rounds:
                raise AssertionError('default_rounds must be >= min_rounds')
            if cls.default_rounds > cls.max_rounds:
                raise AssertionError('default_rounds must be <= max_rounds')
        if cls.rounds_cost not in rounds_cost_values:
            raise AssertionError('unknown rounds cost constant: %r' % (cls.rounds_cost,))

    def test_21_min_rounds(self):
        """test hash() / genconfig() honors min_rounds"""
        self.require_rounds_info()
        handler = self.handler
        min_rounds = handler.min_rounds
        self.do_genconfig(rounds=min_rounds)
        self.do_encrypt('stub', rounds=min_rounds)
        self.assertRaises(ValueError, self.do_genconfig, rounds=min_rounds - 1)
        self.assertRaises(ValueError, self.do_encrypt, 'stub', rounds=min_rounds - 1)

    def test_21b_max_rounds(self):
        """test hash() / genconfig() honors max_rounds"""
        self.require_rounds_info()
        handler = self.handler
        max_rounds = handler.max_rounds
        if max_rounds is not None:
            self.assertRaises(ValueError, self.do_genconfig, rounds=max_rounds + 1)
            self.assertRaises(ValueError, self.do_encrypt, 'stub', rounds=max_rounds + 1)
        if max_rounds is None:
            self.do_stub_encrypt(rounds=(1 << 31) - 1)
        else:
            self.do_stub_encrypt(rounds=max_rounds)

    def _create_using_rounds_helper(self):
        """
        setup test helpers for testing handler.using()'s rounds parameters.
        """
        self.require_rounds_info()
        handler = self.handler
        if handler.name == 'bsdi_crypt':
            orig_handler = handler
            handler = handler.using()
            handler._generate_rounds = classmethod(lambda cls: super(orig_handler, cls)._generate_rounds())
        orig_min_rounds = handler.min_rounds
        orig_max_rounds = handler.max_rounds
        orig_default_rounds = handler.default_rounds
        medium = ((orig_max_rounds or 9999) + orig_min_rounds) // 2
        if medium == orig_default_rounds:
            medium += 1
        small = (orig_min_rounds + medium) // 2
        large = ((orig_max_rounds or 9999) + medium) // 2
        if handler.name == 'bsdi_crypt':
            small |= 1
            medium |= 1
            large |= 1
            adj = 2
        else:
            adj = 1
        with self.assertWarningList([]):
            subcls = handler.using(min_desired_rounds=small, max_desired_rounds=large, default_rounds=medium)
        return (handler, subcls, small, medium, large, adj)

    def test_has_rounds_using_harness(self):
        """
        HasRounds.using() -- sanity check test harness
        """
        self.require_rounds_info()
        handler = self.handler
        orig_min_rounds = handler.min_rounds
        orig_max_rounds = handler.max_rounds
        orig_default_rounds = handler.default_rounds
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
        self.assertEqual(handler.min_rounds, orig_min_rounds)
        self.assertEqual(handler.max_rounds, orig_max_rounds)
        self.assertEqual(handler.min_desired_rounds, None)
        self.assertEqual(handler.max_desired_rounds, None)
        self.assertEqual(handler.default_rounds, orig_default_rounds)
        self.assertEqual(subcls.min_rounds, orig_min_rounds)
        self.assertEqual(subcls.max_rounds, orig_max_rounds)
        self.assertEqual(subcls.default_rounds, medium)
        self.assertEqual(subcls.min_desired_rounds, small)
        self.assertEqual(subcls.max_desired_rounds, large)

    def test_has_rounds_using_w_min_rounds(self):
        """
        HasRounds.using() -- min_rounds / min_desired_rounds
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
        orig_min_rounds = handler.min_rounds
        orig_max_rounds = handler.max_rounds
        orig_default_rounds = handler.default_rounds
        if orig_min_rounds > 0:
            self.assertRaises(ValueError, handler.using, min_desired_rounds=orig_min_rounds - adj)
            with self.assertWarningList([PasslibHashWarning]):
                temp = handler.using(min_desired_rounds=orig_min_rounds - adj, relaxed=True)
            self.assertEqual(temp.min_desired_rounds, orig_min_rounds)
        if orig_max_rounds:
            self.assertRaises(ValueError, handler.using, min_desired_rounds=orig_max_rounds + adj)
            with self.assertWarningList([PasslibHashWarning]):
                temp = handler.using(min_desired_rounds=orig_max_rounds + adj, relaxed=True)
            self.assertEqual(temp.min_desired_rounds, orig_max_rounds)
        with self.assertWarningList([]):
            temp = subcls.using(min_desired_rounds=small - adj)
        self.assertEqual(temp.min_desired_rounds, small - adj)
        temp = subcls.using(min_desired_rounds=small + 2 * adj)
        self.assertEqual(temp.min_desired_rounds, small + 2 * adj)
        with self.assertWarningList([]):
            temp = subcls.using(min_desired_rounds=large + adj)
        self.assertEqual(temp.min_desired_rounds, large + adj)
        self.assertEqual(get_effective_rounds(subcls, small + adj), small + adj)
        self.assertEqual(get_effective_rounds(subcls, small), small)
        with self.assertWarningList([]):
            self.assertEqual(get_effective_rounds(subcls, small - adj), small - adj)
        temp = handler.using(min_rounds=small)
        self.assertEqual(temp.min_desired_rounds, small)
        temp = handler.using(min_rounds=str(small))
        self.assertEqual(temp.min_desired_rounds, small)
        self.assertRaises(ValueError, handler.using, min_rounds=str(small) + 'xxx')

    def test_has_rounds_replace_w_max_rounds(self):
        """
        HasRounds.using() -- max_rounds / max_desired_rounds
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
        orig_min_rounds = handler.min_rounds
        orig_max_rounds = handler.max_rounds
        if orig_min_rounds > 0:
            self.assertRaises(ValueError, handler.using, max_desired_rounds=orig_min_rounds - adj)
            with self.assertWarningList([PasslibHashWarning]):
                temp = handler.using(max_desired_rounds=orig_min_rounds - adj, relaxed=True)
            self.assertEqual(temp.max_desired_rounds, orig_min_rounds)
        if orig_max_rounds:
            self.assertRaises(ValueError, handler.using, max_desired_rounds=orig_max_rounds + adj)
            with self.assertWarningList([PasslibHashWarning]):
                temp = handler.using(max_desired_rounds=orig_max_rounds + adj, relaxed=True)
            self.assertEqual(temp.max_desired_rounds, orig_max_rounds)
        with self.assertWarningList([PasslibConfigWarning]):
            temp = subcls.using(max_desired_rounds=small - adj)
        self.assertEqual(temp.max_desired_rounds, small)
        self.assertRaises(ValueError, subcls.using, min_desired_rounds=medium + adj, max_desired_rounds=medium - adj)
        temp = subcls.using(min_desired_rounds=large - 2 * adj)
        self.assertEqual(temp.min_desired_rounds, large - 2 * adj)
        with self.assertWarningList([]):
            temp = subcls.using(max_desired_rounds=large + adj)
        self.assertEqual(temp.max_desired_rounds, large + adj)
        self.assertEqual(get_effective_rounds(subcls, large - adj), large - adj)
        self.assertEqual(get_effective_rounds(subcls, large), large)
        with self.assertWarningList([]):
            self.assertEqual(get_effective_rounds(subcls, large + adj), large + adj)
        temp = handler.using(max_rounds=large)
        self.assertEqual(temp.max_desired_rounds, large)
        temp = handler.using(max_desired_rounds=str(large))
        self.assertEqual(temp.max_desired_rounds, large)
        self.assertRaises(ValueError, handler.using, max_desired_rounds=str(large) + 'xxx')

    def test_has_rounds_using_w_default_rounds(self):
        """
        HasRounds.using() -- default_rounds
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
        orig_max_rounds = handler.max_rounds
        temp = subcls.using(min_rounds=medium + adj)
        self.assertEqual(temp.default_rounds, medium + adj)
        temp = subcls.using(max_rounds=medium - adj)
        self.assertEqual(temp.default_rounds, medium - adj)
        self.assertRaises(ValueError, subcls.using, default_rounds=small - adj)
        if orig_max_rounds:
            self.assertRaises(ValueError, subcls.using, default_rounds=large + adj)
        self.assertEqual(get_effective_rounds(subcls), medium)
        self.assertEqual(get_effective_rounds(subcls, medium + adj), medium + adj)
        temp = handler.using(default_rounds=str(medium))
        self.assertEqual(temp.default_rounds, medium)
        self.assertRaises(ValueError, handler.using, default_rounds=str(medium) + 'xxx')

    def test_has_rounds_using_w_rounds(self):
        """
        HasRounds.using() -- rounds
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
        orig_max_rounds = handler.max_rounds
        temp = subcls.using(rounds=medium + adj)
        self.assertEqual(temp.min_desired_rounds, medium + adj)
        self.assertEqual(temp.default_rounds, medium + adj)
        self.assertEqual(temp.max_desired_rounds, medium + adj)
        temp = subcls.using(rounds=medium + 1, min_rounds=small + adj, default_rounds=medium, max_rounds=large - adj)
        self.assertEqual(temp.min_desired_rounds, small + adj)
        self.assertEqual(temp.default_rounds, medium)
        self.assertEqual(temp.max_desired_rounds, large - adj)

    def test_has_rounds_using_w_vary_rounds_parsing(self):
        """
        HasRounds.using() -- vary_rounds parsing
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()

        def parse(value):
            return subcls.using(vary_rounds=value).vary_rounds
        self.assertEqual(parse(0.1), 0.1)
        self.assertEqual(parse('0.1'), 0.1)
        self.assertEqual(parse('10%'), 0.1)
        self.assertEqual(parse(1000), 1000)
        self.assertEqual(parse('1000'), 1000)
        self.assertRaises(ValueError, parse, -0.1)
        self.assertRaises(ValueError, parse, 1.1)

    def test_has_rounds_using_w_vary_rounds_generation(self):
        """
        HasRounds.using() -- vary_rounds generation
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()

        def get_effective_range(cls):
            seen = set((get_effective_rounds(cls) for _ in irange(1000)))
            return (min(seen), max(seen))

        def assert_rounds_range(vary_rounds, lower, upper):
            temp = subcls.using(vary_rounds=vary_rounds)
            seen_lower, seen_upper = get_effective_range(temp)
            self.assertEqual(seen_lower, lower, 'vary_rounds had wrong lower limit:')
            self.assertEqual(seen_upper, upper, 'vary_rounds had wrong upper limit:')
        assert_rounds_range(0, medium, medium)
        assert_rounds_range('0%', medium, medium)
        assert_rounds_range(adj, medium - adj, medium + adj)
        assert_rounds_range(50, max(small, medium - 50), min(large, medium + 50))
        if handler.rounds_cost == 'log2':
            assert_rounds_range('1%', medium, medium)
            assert_rounds_range('49%', medium, medium)
            assert_rounds_range('50%', medium - adj, medium)
        else:
            lower, upper = get_effective_range(subcls.using(vary_rounds='50%'))
            self.assertGreaterEqual(lower, max(small, medium * 0.5))
            self.assertLessEqual(lower, max(small, medium * 0.8))
            self.assertGreaterEqual(upper, min(large, medium * 1.2))
            self.assertLessEqual(upper, min(large, medium * 1.5))

    def test_has_rounds_using_and_needs_update(self):
        """
        HasRounds.using() -- desired_rounds + needs_update()
        """
        handler, subcls, small, medium, large, adj = self._create_using_rounds_helper()
        temp = subcls.using(min_desired_rounds=small + 2, max_desired_rounds=large - 2)
        small_hash = self.do_stub_encrypt(subcls, rounds=small)
        medium_hash = self.do_stub_encrypt(subcls, rounds=medium)
        large_hash = self.do_stub_encrypt(subcls, rounds=large)
        self.assertFalse(subcls.needs_update(small_hash))
        self.assertFalse(subcls.needs_update(medium_hash))
        self.assertFalse(subcls.needs_update(large_hash))
        self.assertTrue(temp.needs_update(small_hash))
        self.assertFalse(temp.needs_update(medium_hash))
        self.assertTrue(temp.needs_update(large_hash))

    def require_many_idents(self):
        handler = self.handler
        if not isinstance(handler, type) or not issubclass(handler, uh.HasManyIdents):
            raise self.skipTest("handler doesn't derive from HasManyIdents")

    def test_30_HasManyIdents(self):
        """validate HasManyIdents configuration"""
        cls = self.handler
        self.require_many_idents()
        self.assertTrue('ident' in cls.setting_kwds)
        for value in cls.ident_values:
            self.assertIsInstance(value, unicode, 'cls.ident_values must be unicode:')
        self.assertTrue(len(cls.ident_values) > 1, 'cls.ident_values must have 2+ elements:')
        self.assertIsInstance(cls.default_ident, unicode, 'cls.default_ident must be unicode:')
        self.assertTrue(cls.default_ident in cls.ident_values, 'cls.default_ident must specify member of cls.ident_values')
        if cls.ident_aliases:
            for alias, ident in iteritems(cls.ident_aliases):
                self.assertIsInstance(alias, unicode, 'cls.ident_aliases keys must be unicode:')
                self.assertIsInstance(ident, unicode, 'cls.ident_aliases values must be unicode:')
                self.assertTrue(ident in cls.ident_values, 'cls.ident_aliases must map to cls.ident_values members: %r' % (ident,))
        handler = cls
        hash = self.get_sample_hash()[1]
        kwds = handler.parsehash(hash)
        del kwds['ident']
        handler(ident=cls.default_ident, **kwds)
        self.assertRaises(TypeError, handler, **kwds)
        handler(use_defaults=True, **kwds)
        self.assertRaises(ValueError, handler, ident='xXx', **kwds)

    def test_has_many_idents_using(self):
        """HasManyIdents.using() -- 'default_ident' and 'ident' keywords"""
        self.require_many_idents()
        handler = self.handler
        orig_ident = handler.default_ident
        for alt_ident in handler.ident_values:
            if alt_ident != orig_ident:
                break
        else:
            raise AssertionError('expected to find alternate ident: default=%r values=%r' % (orig_ident, handler.ident_values))

        def effective_ident(cls):
            cls = unwrap_handler(cls)
            return cls(use_defaults=True).ident
        subcls = handler.using()
        self.assertEqual(subcls.default_ident, orig_ident)
        subcls = handler.using(default_ident=alt_ident)
        self.assertEqual(subcls.default_ident, alt_ident)
        self.assertEqual(handler.default_ident, orig_ident)
        self.assertEqual(effective_ident(subcls), alt_ident)
        self.assertEqual(effective_ident(handler), orig_ident)
        self.assertRaises(ValueError, handler.using, default_ident='xXx')
        subcls = handler.using(ident=alt_ident)
        self.assertEqual(subcls.default_ident, alt_ident)
        self.assertEqual(handler.default_ident, orig_ident)
        self.assertRaises(TypeError, handler.using, default_ident=alt_ident, ident=alt_ident)
        if handler.ident_aliases:
            for alias, ident in handler.ident_aliases.items():
                subcls = handler.using(ident=alias)
                self.assertEqual(subcls.default_ident, ident, msg='alias %r:' % alias)

    def test_truncate_error_setting(self):
        """
        validate 'truncate_error' setting & related attributes
        """
        hasher = self.handler
        if hasher.truncate_size is None:
            self.assertNotIn('truncate_error', hasher.setting_kwds)
            return
        if not hasher.truncate_error:
            self.assertFalse(hasher.truncate_verify_reject)
        if 'truncate_error' not in hasher.setting_kwds:
            self.assertTrue(hasher.truncate_error)
            return

        def parse_value(value):
            return hasher.using(truncate_error=value).truncate_error
        self.assertEqual(parse_value(None), hasher.truncate_error)
        self.assertEqual(parse_value(True), True)
        self.assertEqual(parse_value('true'), True)
        self.assertEqual(parse_value(False), False)
        self.assertEqual(parse_value('false'), False)
        self.assertRaises(ValueError, parse_value, 'xxx')

    def test_secret_wo_truncate_size(self):
        """
        test no password size limits enforced (if truncate_size=None)
        """
        hasher = self.handler
        if hasher.truncate_size is not None:
            self.assertGreaterEqual(hasher.truncate_size, 1)
            raise self.skipTest('truncate_size is set')
        secret = 'too many secrets' * 16
        alt = 'x'
        hash = self.do_encrypt(secret)
        verify_success = not hasher.is_disabled
        self.assertEqual(self.do_verify(secret, hash), verify_success, msg='verify rejected correct secret')
        alt_secret = secret[:-1] + alt
        self.assertFalse(self.do_verify(alt_secret, hash), 'full password not used in digest')

    def test_secret_w_truncate_size(self):
        """
        test password size limits raise truncate_error (if appropriate)
        """
        handler = self.handler
        truncate_size = handler.truncate_size
        if not truncate_size:
            raise self.skipTest('truncate_size not set')
        size_error_type = exc.PasswordSizeError
        if 'truncate_error' in handler.setting_kwds:
            without_error = handler.using(truncate_error=False)
            with_error = handler.using(truncate_error=True)
            size_error_type = exc.PasswordTruncateError
        elif handler.truncate_error:
            without_error = None
            with_error = handler
        else:
            without_error = handler
            with_error = None
        base = 'too many secrets'
        alt = 'x'
        long_secret = repeat_string(base, truncate_size + 1)
        short_secret = long_secret[:-1]
        alt_long_secret = long_secret[:-1] + alt
        alt_short_secret = short_secret[:-1] + alt
        short_verify_success = not handler.is_disabled
        long_verify_success = short_verify_success and (not handler.truncate_verify_reject)
        assert without_error or with_error
        for cand_hasher in [without_error, with_error]:
            short_hash = self.do_encrypt(short_secret, handler=cand_hasher)
            self.assertEqual(self.do_verify(short_secret, short_hash, handler=cand_hasher), short_verify_success)
            self.assertFalse(self.do_verify(alt_short_secret, short_hash, handler=with_error), 'truncate_size value is too large')
            self.assertEqual(self.do_verify(long_secret, short_hash, handler=cand_hasher), long_verify_success)
        if without_error:
            long_hash = self.do_encrypt(long_secret, handler=without_error)
            self.assertEqual(self.do_verify(long_secret, long_hash, handler=without_error), short_verify_success)
            self.assertEqual(self.do_verify(alt_long_secret, long_hash, handler=without_error), short_verify_success)
            self.assertTrue(self.do_verify(short_secret, long_hash, handler=without_error))
        if with_error:
            err = self.assertRaises(size_error_type, self.do_encrypt, long_secret, handler=with_error)
            self.assertEqual(err.max_size, truncate_size)

    def test_61_secret_case_sensitive(self):
        """test password case sensitivity"""
        hash_insensitive = self.secret_case_insensitive is True
        verify_insensitive = self.secret_case_insensitive in [True, 'verify-only']
        lower = 'test'
        upper = 'TEST'
        h1 = self.do_encrypt(lower)
        if verify_insensitive and (not self.handler.is_disabled):
            self.assertTrue(self.do_verify(upper, h1), 'verify() should not be case sensitive')
        else:
            self.assertFalse(self.do_verify(upper, h1), 'verify() should be case sensitive')
        h2 = self.do_encrypt(upper)
        if verify_insensitive and (not self.handler.is_disabled):
            self.assertTrue(self.do_verify(lower, h2), 'verify() should not be case sensitive')
        else:
            self.assertFalse(self.do_verify(lower, h2), 'verify() should be case sensitive')
        h2 = self.do_genhash(upper, h1)
        if hash_insensitive or (self.handler.is_disabled and (not self.disabled_contains_salt)):
            self.assertEqual(h2, h1, 'genhash() should not be case sensitive')
        else:
            self.assertNotEqual(h2, h1, 'genhash() should be case sensitive')

    def test_62_secret_border(self):
        """test non-string passwords are rejected"""
        hash = self.get_sample_hash()[1]
        self.assertRaises(TypeError, self.do_encrypt, None)
        self.assertRaises(TypeError, self.do_genhash, None, hash)
        self.assertRaises(TypeError, self.do_verify, None, hash)
        self.assertRaises(TypeError, self.do_encrypt, 1)
        self.assertRaises(TypeError, self.do_genhash, 1, hash)
        self.assertRaises(TypeError, self.do_verify, 1, hash)

    def test_63_large_secret(self):
        """test MAX_PASSWORD_SIZE is enforced"""
        from passlib.exc import PasswordSizeError
        from passlib.utils import MAX_PASSWORD_SIZE
        secret = '.' * (1 + MAX_PASSWORD_SIZE)
        hash = self.get_sample_hash()[1]
        err = self.assertRaises(PasswordSizeError, self.do_genhash, secret, hash)
        self.assertEqual(err.max_size, MAX_PASSWORD_SIZE)
        self.assertRaises(PasswordSizeError, self.do_encrypt, secret)
        self.assertRaises(PasswordSizeError, self.do_verify, secret, hash)

    def test_64_forbidden_chars(self):
        """test forbidden characters not allowed in password"""
        chars = self.forbidden_characters
        if not chars:
            raise self.skipTest('none listed')
        base = u('stub')
        if isinstance(chars, bytes):
            from passlib.utils.compat import iter_byte_chars
            chars = iter_byte_chars(chars)
            base = base.encode('ascii')
        for c in chars:
            self.assertRaises(ValueError, self.do_encrypt, base + c + base)

    def is_secret_8bit(self, secret):
        secret = self.populate_context(secret, {})
        return not is_ascii_safe(secret)

    def expect_os_crypt_failure(self, secret):
        """
        check if we're expecting potential verify failure due to crypt.crypt() encoding limitation
        """
        if PY3 and self.backend == 'os_crypt' and isinstance(secret, bytes):
            try:
                secret.decode('utf-8')
            except UnicodeDecodeError:
                return True
        return False

    def test_70_hashes(self):
        """test known hashes"""
        self.assertTrue(self.known_correct_hashes or self.known_correct_configs, "test must set at least one of 'known_correct_hashes' or 'known_correct_configs'")
        saw8bit = False
        for secret, hash in self.iter_known_hashes():
            if self.is_secret_8bit(secret):
                saw8bit = True
            self.assertTrue(self.do_identify(hash), 'identify() failed to identify hash: %r' % (hash,))
            expect_os_crypt_failure = self.expect_os_crypt_failure(secret)
            try:
                self.check_verify(secret, hash, 'verify() of known hash failed: secret=%r, hash=%r' % (secret, hash))
                result = self.do_genhash(secret, hash)
                self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
                if self.handler.is_disabled and self.disabled_contains_salt:
                    continue
                self.assertEqual(result, hash, 'genhash() failed to reproduce known hash: secret=%r, hash=%r: result=%r' % (secret, hash, result))
            except MissingBackendError:
                if not expect_os_crypt_failure:
                    raise
        if not saw8bit:
            warn('%s: no 8-bit secrets tested' % self.__class__)

    def test_71_alternates(self):
        """test known alternate hashes"""
        if not self.known_alternate_hashes:
            raise self.skipTest('no alternate hashes provided')
        for alt, secret, hash in self.known_alternate_hashes:
            self.assertTrue(self.do_identify(hash), 'identify() failed to identify alternate hash: %r' % (hash,))
            self.check_verify(secret, alt, 'verify() of known alternate hash failed: secret=%r, hash=%r' % (secret, alt))
            result = self.do_genhash(secret, alt)
            self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
            if self.handler.is_disabled and self.disabled_contains_salt:
                continue
            self.assertEqual(result, hash, 'genhash() failed to normalize known alternate hash: secret=%r, alt=%r, hash=%r: result=%r' % (secret, alt, hash, result))

    def test_72_configs(self):
        """test known config strings"""
        if not self.handler.setting_kwds:
            self.assertFalse(self.known_correct_configs, 'handler should not have config strings')
            raise self.skipTest('hash has no settings')
        if not self.known_correct_configs:
            raise self.skipTest('no config strings provided')
        if self.filter_config_warnings:
            warnings.filterwarnings('ignore', category=PasslibHashWarning)
        for config, secret, hash in self.known_correct_configs:
            self.assertTrue(self.do_identify(config), 'identify() failed to identify known config string: %r' % (config,))
            self.assertRaises(ValueError, self.do_verify, secret, config, __msg__='verify() failed to reject config string: %r' % (config,))
            result = self.do_genhash(secret, config)
            self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
            self.assertEqual(result, hash, 'genhash() failed to reproduce known hash from config: secret=%r, config=%r, hash=%r: result=%r' % (secret, config, hash, result))

    def test_73_unidentified(self):
        """test known unidentifiably-mangled strings"""
        if not self.known_unidentified_hashes:
            raise self.skipTest('no unidentified hashes provided')
        for hash in self.known_unidentified_hashes:
            self.assertFalse(self.do_identify(hash), 'identify() incorrectly identified known unidentifiable hash: %r' % (hash,))
            self.assertRaises(ValueError, self.do_verify, 'stub', hash, __msg__='verify() failed to throw error for unidentifiable hash: %r' % (hash,))
            self.assertRaises(ValueError, self.do_genhash, 'stub', hash, __msg__='genhash() failed to throw error for unidentifiable hash: %r' % (hash,))

    def test_74_malformed(self):
        """test known identifiable-but-malformed strings"""
        if not self.known_malformed_hashes:
            raise self.skipTest('no malformed hashes provided')
        for hash in self.known_malformed_hashes:
            self.assertTrue(self.do_identify(hash), 'identify() failed to identify known malformed hash: %r' % (hash,))
            self.assertRaises(ValueError, self.do_verify, 'stub', hash, __msg__='verify() failed to throw error for malformed hash: %r' % (hash,))
            self.assertRaises(ValueError, self.do_genhash, 'stub', hash, __msg__='genhash() failed to throw error for malformed hash: %r' % (hash,))

    def test_75_foreign(self):
        """test known foreign hashes"""
        if self.accepts_all_hashes:
            raise self.skipTest('not applicable')
        if not self.known_other_hashes:
            raise self.skipTest('no foreign hashes provided')
        for name, hash in self.known_other_hashes:
            if name == self.handler.name:
                self.assertTrue(self.do_identify(hash), 'identify() failed to identify known hash: %r' % (hash,))
                self.do_verify('stub', hash)
                result = self.do_genhash('stub', hash)
                self.assertIsInstance(result, str, 'genhash() failed to return native string: %r' % (result,))
            else:
                self.assertFalse(self.do_identify(hash), 'identify() incorrectly identified hash belonging to %s: %r' % (name, hash))
                self.assertRaises(ValueError, self.do_verify, 'stub', hash, __msg__='verify() failed to throw error for hash belonging to %s: %r' % (name, hash))
                self.assertRaises(ValueError, self.do_genhash, 'stub', hash, __msg__='genhash() failed to throw error for hash belonging to %s: %r' % (name, hash))

    def test_76_hash_border(self):
        """test non-string hashes are rejected"""
        self.assertRaises(TypeError, self.do_identify, None)
        self.assertRaises(TypeError, self.do_verify, 'stub', None)
        self.assertRaises(TypeError, self.do_genhash, 'stub', None)
        self.assertRaises(TypeError, self.do_identify, 1)
        self.assertRaises(TypeError, self.do_verify, 'stub', 1)
        self.assertRaises(TypeError, self.do_genhash, 'stub', 1)
        for hash in [u(''), b'']:
            if self.accepts_all_hashes:
                self.assertTrue(self.do_identify(hash))
                self.do_verify('stub', hash)
                result = self.do_genhash('stub', hash)
                self.check_returned_native_str(result, 'genhash')
            else:
                self.assertFalse(self.do_identify(hash), 'identify() incorrectly identified empty hash')
                self.assertRaises(ValueError, self.do_verify, 'stub', hash, __msg__='verify() failed to reject empty hash')
                self.assertRaises(ValueError, self.do_genhash, 'stub', hash, __msg__='genhash() failed to reject empty hash')
        self.do_identify('â\x82¬Â¥$')
        self.do_identify('abc\x91\x00')
    known_parsehash_results = []

    def require_parsehash(self):
        if not hasattr(self.handler, 'parsehash'):
            raise SkipTest('parsehash() not implemented')

    def test_70_parsehash(self):
        """
        parsehash()
        """
        self.require_parsehash()
        handler = self.handler
        hash = self.do_encrypt('stub')
        result = handler.parsehash(hash)
        self.assertIsInstance(result, dict)
        result2 = handler.parsehash(hash, checksum=False)
        correct2 = result.copy()
        correct2.pop('checksum', None)
        self.assertEqual(result2, correct2)
        result3 = handler.parsehash(hash, sanitize=True)
        correct3 = result.copy()
        if PY2:
            warnings.filterwarnings('ignore', '.*unequal comparison failed to convert.*', category=UnicodeWarning)
        for key in ('salt', 'checksum'):
            if key in result3:
                self.assertNotEqual(result3[key], correct3[key])
                self.assert_is_masked(result3[key])
                correct3[key] = result3[key]
        self.assertEqual(result3, correct3)

    def assert_is_masked(self, value):
        """
        check value properly masked by :func:`passlib.utils.mask_value`
        """
        if value is None:
            return
        self.assertIsInstance(value, unicode)
        ref = value if len(value) < 8 else value[4:]
        if set(ref) == set(['*']):
            return True
        raise self.fail('value not masked: %r' % value)

    def test_71_parsehash_results(self):
        """
        parsehash() -- known outputs
        """
        self.require_parsehash()
        samples = self.known_parsehash_results
        if not samples:
            raise self.skipTest('no samples present')
        for hash, correct in self.known_parsehash_results:
            result = self.handler.parsehash(hash)
            self.assertEqual(result, correct, 'hash=%r:' % hash)

    def test_77_fuzz_input(self, threaded=False):
        """fuzz testing -- random passwords and options

        This test attempts to perform some basic fuzz testing of the hash,
        based on whatever information can be found about it.
        It does as much as it can within a fixed amount of time
        (defaults to 1 second, but can be overridden via $PASSLIB_TEST_FUZZ_TIME).
        It tests the following:

        * randomly generated passwords including extended unicode chars
        * randomly selected rounds values (if rounds supported)
        * randomly selected salt sizes (if salts supported)
        * randomly selected identifiers (if multiple found)
        * runs output of selected backend against other available backends
          (if any) to detect errors occurring between different backends.
        * runs output against other "external" verifiers such as OS crypt()

        :param report_thread_state:
            if true, writes state of loop to current_thread().passlib_fuzz_state.
            used to help debug multi-threaded fuzz test issues (below)
        """
        if self.handler.is_disabled:
            raise self.skipTest('not applicable')
        from passlib.utils import tick
        max_time = self.max_fuzz_time
        if max_time <= 0:
            raise self.skipTest('disabled by test mode')
        verifiers = self.get_fuzz_verifiers(threaded=threaded)

        def vname(v):
            return (v.__doc__ or v.__name__).splitlines()[0]
        if threaded:
            thread_name = threading.current_thread().name
        else:
            thread_name = 'fuzz test'
        rng = self.getRandom(name=thread_name)
        generator = self.FuzzHashGenerator(self, rng)
        log.debug('%s: %s: started; max_time=%r verifiers=%d (%s)', self.descriptionPrefix, thread_name, max_time, len(verifiers), ', '.join((vname(v) for v in verifiers)))
        start = tick()
        stop = start + max_time
        count = 0
        while tick() <= stop:
            opts = generator.generate()
            secret = opts['secret']
            other = opts['other']
            settings = opts['settings']
            ctx = opts['context']
            if ctx:
                settings['context'] = ctx
            hash = self.do_encrypt(secret, **settings)
            for verify in verifiers:
                name = vname(verify)
                result = verify(secret, hash, **ctx)
                if result == 'skip':
                    continue
                assert result is True or result is False
                if not result:
                    raise self.failureException('failed to verify against %r verifier: secret=%r config=%r hash=%r' % (name, secret, settings, hash))
                if rng.random() < 0.1:
                    result = verify(other, hash, **ctx)
                    if result and result != 'skip':
                        raise self.failureException('was able to verify wrong password using %s: wrong_secret=%r real_secret=%r config=%r hash=%r' % (name, other, secret, settings, hash))
            count += 1
        log.debug('%s: %s: done; elapsed=%r count=%r', self.descriptionPrefix, thread_name, tick() - start, count)

    def test_78_fuzz_threading(self):
        """multithreaded fuzz testing -- random password & options using multiple threads

        run test_77 simultaneously in multiple threads
        in an attempt to detect any concurrency issues
        (e.g. the bug fixed by pybcrypt 0.3)
        """
        self.require_TEST_MODE('full')
        import threading
        if self.handler.is_disabled:
            raise self.skipTest('not applicable')
        thread_count = self.fuzz_thread_count
        if thread_count < 1 or self.max_fuzz_time <= 0:
            raise self.skipTest('disabled by test mode')
        failed_lock = threading.Lock()
        failed = [0]

        def wrapper():
            try:
                self.test_77_fuzz_input(threaded=True)
            except SkipTest:
                pass
            except:
                with failed_lock:
                    failed[0] += 1
                raise

        def launch(n):
            cls = type(self)
            name = "Fuzz-Thread-%d ('%s:%s.%s')" % (n, cls.__module__, cls.__name__, self._testMethodName)
            thread = threading.Thread(target=wrapper, name=name)
            thread.setDaemon(True)
            thread.start()
            return thread
        threads = [launch(n) for n in irange(thread_count)]
        timeout = self.max_fuzz_time * thread_count * 4
        stalled = 0
        for thread in threads:
            thread.join(timeout)
            if not thread.is_alive():
                continue
            log.error('%s timed out after %f seconds', thread.name, timeout)
            stalled += 1
        if failed[0]:
            raise self.fail('%d/%d threads failed concurrent fuzz testing (see error log for details)' % (failed[0], thread_count))
        if stalled:
            raise self.fail('%d/%d threads stalled during concurrent fuzz testing (see error log for details)' % (stalled, thread_count))

    @property
    def max_fuzz_time(self):
        """amount of time to spend on fuzz testing"""
        value = float(os.environ.get('PASSLIB_TEST_FUZZ_TIME') or 0)
        if value:
            return value
        elif TEST_MODE(max='quick'):
            return 0
        elif TEST_MODE(max='default'):
            return 1
        else:
            return 5

    @property
    def fuzz_thread_count(self):
        """number of threads for threaded fuzz testing"""
        value = int(os.environ.get('PASSLIB_TEST_FUZZ_THREADS') or 0)
        if value:
            return value
        elif TEST_MODE(max='quick'):
            return 0
        else:
            return 10
    fuzz_verifiers = ('fuzz_verifier_default',)

    def get_fuzz_verifiers(self, threaded=False):
        """return list of password verifiers (including external libs)

        used by fuzz testing.
        verifiers should be callable with signature
        ``func(password: unicode, hash: ascii str) -> ok: bool``.
        """
        handler = self.handler
        verifiers = []
        for method_name in self.fuzz_verifiers:
            func = getattr(self, method_name)()
            if func is not None:
                verifiers.append(func)
        if hasattr(handler, 'backends') and TEST_MODE('full') and (not threaded):

            def maker(backend):

                def func(secret, hash):
                    orig_backend = handler.get_backend()
                    try:
                        handler.set_backend(backend)
                        return handler.verify(secret, hash)
                    finally:
                        handler.set_backend(orig_backend)
                func.__name__ = 'check_' + backend + '_backend'
                func.__doc__ = backend + '-backend'
                return func
            for backend in iter_alt_backends(handler):
                verifiers.append(maker(backend))
        return verifiers

    def fuzz_verifier_default(self):

        def check_default(secret, hash, **ctx):
            return self.do_verify(secret, hash, **ctx)
        if self.backend:
            check_default.__doc__ = self.backend + '-backend'
        else:
            check_default.__doc__ = 'self'
        return check_default

    class FuzzHashGenerator(object):
        """
        helper which takes care of generating random
        passwords & configuration options to test hash with.
        separate from test class so we can create one per thread.
        """
        password_alphabet = u('qwertyASDF1234<>.@*#! áəБℓ')
        password_encoding = 'utf-8'
        settings_map = dict(rounds='random_rounds', salt_size='random_salt_size', ident='random_ident')
        context_map = {}

        def __init__(self, test, rng):
            self.test = test
            self.handler = test.handler
            self.rng = rng

        def generate(self):
            """
            generate random password and options for fuzz testing.
            :returns:
                `(secret, other_secret, settings_kwds, context_kwds)`
            """

            def gendict(map):
                out = {}
                for key, meth in map.items():
                    value = getattr(self, meth)()
                    if value is not None:
                        out[key] = value
                return out
            secret, other = self.random_password_pair()
            return dict(secret=secret, other=other, settings=gendict(self.settings_map), context=gendict(self.context_map))

        def randintgauss(self, lower, upper, mu, sigma):
            """generate random int w/ gauss distirbution"""
            value = self.rng.normalvariate(mu, sigma)
            return int(limit(value, lower, upper))

        def random_rounds(self):
            handler = self.handler
            if not has_rounds_info(handler):
                return None
            default = handler.default_rounds or handler.min_rounds
            lower = handler.min_rounds
            if handler.rounds_cost == 'log2':
                upper = default
            else:
                upper = min(default * 2, handler.max_rounds)
            return self.randintgauss(lower, upper, default, default * 0.5)

        def random_salt_size(self):
            handler = self.handler
            if not (has_salt_info(handler) and 'salt_size' in handler.setting_kwds):
                return None
            default = handler.default_salt_size
            lower = handler.min_salt_size
            upper = handler.max_salt_size or default * 4
            return self.randintgauss(lower, upper, default, default * 0.5)

        def random_ident(self):
            rng = self.rng
            handler = self.handler
            if 'ident' not in handler.setting_kwds or not hasattr(handler, 'ident_values'):
                return None
            if rng.random() < 0.5:
                return None
            handler = getattr(handler, 'wrapped', handler)
            return rng.choice(handler.ident_values)

        def random_password_pair(self):
            """generate random password, and non-matching alternate password"""
            secret = self.random_password()
            while True:
                other = self.random_password()
                if self.accept_password_pair(secret, other):
                    break
            rng = self.rng
            if rng.randint(0, 1):
                secret = secret.encode(self.password_encoding)
            if rng.randint(0, 1):
                other = other.encode(self.password_encoding)
            return (secret, other)

        def random_password(self):
            """generate random passwords for fuzz testing"""
            rng = self.rng
            if rng.random() < 0.0001:
                return u('')
            handler = self.handler
            truncate_size = handler.truncate_error and handler.truncate_size
            max_size = truncate_size or 999999
            if max_size < 50 or rng.random() < 0.5:
                size = self.randintgauss(1, min(max_size, 50), 15, 15)
            else:
                size = self.randintgauss(50, min(max_size, 99), 70, 20)
            result = getrandstr(rng, self.password_alphabet, size)
            if truncate_size and isinstance(result, unicode):
                while len(result.encode('utf-8')) > truncate_size:
                    result = result[:-1]
            return result

        def accept_password_pair(self, secret, other):
            """verify fuzz pair contains different passwords"""
            return secret != other

    def test_disable_and_enable(self):
        """.disable() / .enable() methods"""
        handler = self.handler
        if not handler.is_disabled:
            self.assertFalse(hasattr(handler, 'disable'))
            self.assertFalse(hasattr(handler, 'enable'))
            self.assertFalse(self.disabled_contains_salt)
            raise self.skipTest('not applicable')
        disabled_default = handler.disable()
        self.assertIsInstance(disabled_default, str, msg='disable() must return native string')
        self.assertTrue(handler.identify(disabled_default), msg="identify() didn't recognize disable() result: %r" % disabled_default)
        stub = self.getRandom().choice(self.known_other_hashes)[1]
        disabled_stub = handler.disable(stub)
        self.assertIsInstance(disabled_stub, str, msg='disable() must return native string')
        self.assertTrue(handler.identify(disabled_stub), msg="identify() didn't recognize disable() result: %r" % disabled_stub)
        self.assertRaisesRegex(ValueError, 'cannot restore original hash', handler.enable, disabled_default)
        try:
            result = handler.enable(disabled_stub)
            error = None
        except ValueError as e:
            result = None
            error = e
        if error is None:
            self.assertIsInstance(result, str, msg='enable() must return native string')
            self.assertEqual(result, stub)
        else:
            self.assertIsInstance(error, ValueError)
            self.assertRegex('cannot restore original hash', str(error))
        disabled_default2 = handler.disable()
        if self.disabled_contains_salt:
            self.assertNotEqual(disabled_default2, disabled_default)
        elif error is None:
            self.assertEqual(disabled_default2, disabled_default)
        disabled_stub2 = handler.disable(stub)
        if self.disabled_contains_salt:
            self.assertNotEqual(disabled_stub2, disabled_stub)
        else:
            self.assertEqual(disabled_stub2, disabled_stub)
        disabled_other = handler.disable(stub + 'xxx')
        if self.disabled_contains_salt or error is None:
            self.assertNotEqual(disabled_other, disabled_stub)
        else:
            self.assertEqual(disabled_other, disabled_stub)