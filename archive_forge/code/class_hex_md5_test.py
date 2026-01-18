from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class hex_md5_test(HandlerCase):
    handler = hash.hex_md5
    known_correct_hashes = [('password', '5f4dcc3b5aa765d61d8327deb882cf99'), (UPASS_TABLE, '05473f8a19f66815e737b33264a0d0b0')]

    def test_mock_fips_mode(self):
        """
        if md5 isn't available, a dummy instance should be created.
        (helps on FIPS systems).
        """
        from passlib.exc import UnknownHashError
        from passlib.crypto.digest import lookup_hash, _set_mock_fips_mode
        supported = lookup_hash('md5', required=False).supported
        self.assertEqual(self.handler.supported, supported)
        if supported:
            _set_mock_fips_mode()
            self.addCleanup(_set_mock_fips_mode, False)
        from passlib.handlers.digests import create_hex_hash
        hasher = create_hex_hash('md5', required=False)
        self.assertFalse(hasher.supported)
        ref1 = '5f4dcc3b5aa765d61d8327deb882cf99'
        ref2 = 'xxx'
        self.assertTrue(hasher.identify(ref1))
        self.assertFalse(hasher.identify(ref2))
        pat = "'md5' hash disabled for fips"
        self.assertRaisesRegex(UnknownHashError, pat, hasher.hash, 'password')
        self.assertRaisesRegex(UnknownHashError, pat, hasher.verify, 'password', ref1)