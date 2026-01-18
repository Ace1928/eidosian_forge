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
class OsCryptMixin(HandlerCase):
    """helper used by create_backend_case() which adds additional features
    to test the os_crypt backend.

    * if crypt support is missing, inserts fake crypt support to simulate
      a working safe_crypt, to test passlib's codepath as fully as possible.

    * extra tests to verify non-conformant crypt implementations are handled
      correctly.

    * check that native crypt support is detected correctly for known platforms.
    """
    platform_crypt_support = []
    __unittest_skip = True
    backend = 'os_crypt'
    using_patched_crypt = False

    def setUp(self):
        assert self.backend == 'os_crypt'
        if not self.handler.has_backend('os_crypt'):
            self._patch_safe_crypt()
        super(OsCryptMixin, self).setUp()

    @classmethod
    def _get_safe_crypt_handler_backend(cls):
        """
        return (handler, backend) pair to use for faking crypt.crypt() support for hash.
        backend will be None if none availabe.
        """
        handler = unwrap_handler(cls.handler)
        handler.get_backend()
        alt_backend = get_alt_backend(handler, 'os_crypt')
        return (handler, alt_backend)

    @property
    def has_os_crypt_fallback(self):
        """
        test if there's a fallback handler to test against if os_crypt can't support
        a specified secret (may be explicitly set to False for some subclasses)
        """
        return self._get_safe_crypt_handler_backend()[0] is not None

    def _patch_safe_crypt(self):
        """if crypt() doesn't support current hash alg, this patches
        safe_crypt() so that it transparently uses another one of the handler's
        backends, so that we can go ahead and test as much of code path
        as possible.
        """
        handler, alt_backend = self._get_safe_crypt_handler_backend()
        if not alt_backend:
            raise AssertionError('handler has no available alternate backends!')
        alt_handler = handler.using()
        alt_handler.set_backend(alt_backend)

        def crypt_stub(secret, hash):
            hash = alt_handler.genhash(secret, hash)
            assert isinstance(hash, str)
            return hash
        import passlib.utils as mod
        self.patchAttr(mod, '_crypt', crypt_stub)
        self.using_patched_crypt = True

    @classmethod
    def _get_skip_backend_reason(cls, backend):
        """
        make sure os_crypt backend is tested
        when it's known os_crypt will be faked by _patch_safe_crypt()
        """
        assert backend == 'os_crypt'
        reason = super(OsCryptMixin, cls)._get_skip_backend_reason(backend)
        from passlib.utils import has_crypt
        if reason == cls._BACKEND_NOT_AVAILABLE and has_crypt:
            if TEST_MODE('full') and cls._get_safe_crypt_handler_backend()[1]:
                return None
            else:
                return 'hash not supported by os crypt()'
        return reason

    def _use_mock_crypt(self):
        """
        patch passlib.utils.safe_crypt() so it returns mock value for duration of test.
        returns function whose .return_value controls what's returned.
        this defaults to None.
        """
        import passlib.utils as mod

        def mock_crypt(secret, config):
            if secret == 'test':
                return mock_crypt.__wrapped__(secret, config)
            else:
                return mock_crypt.return_value
        mock_crypt.__wrapped__ = mod._crypt
        mock_crypt.return_value = None
        self.patchAttr(mod, '_crypt', mock_crypt)
        return mock_crypt

    def test_80_faulty_crypt(self):
        """test with faulty crypt()"""
        hash = self.get_sample_hash()[1]
        exc_types = (exc.InternalBackendError,)
        mock_crypt = self._use_mock_crypt()

        def test(value):
            mock_crypt.return_value = value
            self.assertRaises(exc_types, self.do_genhash, 'stub', hash)
            self.assertRaises(exc_types, self.do_encrypt, 'stub')
            self.assertRaises(exc_types, self.do_verify, 'stub', hash)
        test('$x' + hash[2:])
        test(hash[:-1])
        test(hash + 'x')

    def test_81_crypt_fallback(self):
        """test per-call crypt() fallback"""
        mock_crypt = self._use_mock_crypt()
        mock_crypt.return_value = None
        if self.has_os_crypt_fallback:
            h1 = self.do_encrypt('stub')
            h2 = self.do_genhash('stub', h1)
            self.assertEqual(h2, h1)
            self.assertTrue(self.do_verify('stub', h1))
        else:
            from passlib.exc import InternalBackendError as err_type
            hash = self.get_sample_hash()[1]
            self.assertRaises(err_type, self.do_encrypt, 'stub')
            self.assertRaises(err_type, self.do_genhash, 'stub', hash)
            self.assertRaises(err_type, self.do_verify, 'stub', hash)

    @doesnt_require_backend
    def test_82_crypt_support(self):
        """
        test platform-specific crypt() support detection

        NOTE: this is mainly just a sanity check to ensure the runtime
              detection is functioning correctly on some known platforms,
              so that we can feel more confident it'll work right on unknown ones.
        """
        if hasattr(self.handler, 'orig_prefix'):
            raise self.skipTest('not applicable to wrappers')
        using_backend = not self.using_patched_crypt
        name = self.handler.name
        platform = sys.platform
        for pattern, expected in self.platform_crypt_support:
            if re.match(pattern, platform):
                break
        else:
            raise self.skipTest('no data for %r platform (current host support = %r)' % (platform, using_backend))
        if expected is None:
            raise self.skipTest('varied support on %r platform (current host support = %r)' % (platform, using_backend))
        if expected == using_backend:
            pass
        elif expected:
            self.fail('expected %r platform would have native support for %r' % (platform, name))
        else:
            self.fail('did not expect %r platform would have native support for %r' % (platform, name))

    def fuzz_verifier_crypt(self):
        """test results against OS crypt()"""
        handler = self.handler
        if self.using_patched_crypt or hasattr(handler, 'wrapped'):
            return None
        from crypt import crypt
        from passlib.utils import _safe_crypt_lock
        encoding = self.FuzzHashGenerator.password_encoding

        def check_crypt(secret, hash):
            """stdlib-crypt"""
            if not self.crypt_supports_variant(hash):
                return 'skip'
            secret = to_native_str(secret, encoding)
            with _safe_crypt_lock:
                return crypt(secret, hash) == hash
        return check_crypt

    def crypt_supports_variant(self, hash):
        """
        fuzzy_verified_crypt() helper --
        used to determine if os crypt() supports a particular hash variant.
        """
        return True