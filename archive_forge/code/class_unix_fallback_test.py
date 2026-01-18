from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class unix_fallback_test(HandlerCase):
    handler = hash.unix_fallback
    accepts_all_hashes = True
    known_correct_hashes = [('password', '!'), (UPASS_TABLE, '!')]

    def setUp(self):
        super(unix_fallback_test, self).setUp()
        warnings.filterwarnings('ignore', "'unix_fallback' is deprecated")

    def test_90_wildcard(self):
        """test enable_wildcard flag"""
        h = self.handler
        self.assertTrue(h.verify('password', '', enable_wildcard=True))
        self.assertFalse(h.verify('password', ''))
        for c in '!*x':
            self.assertFalse(h.verify('password', c, enable_wildcard=True))
            self.assertFalse(h.verify('password', c))

    def test_91_preserves_existing(self):
        """test preserves existing disabled hash"""
        handler = self.handler
        self.assertEqual(handler.genhash('stub', ''), '!')
        self.assertEqual(handler.hash('stub'), '!')
        self.assertEqual(handler.genhash('stub', '!asd'), '!asd')