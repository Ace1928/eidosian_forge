from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class unix_disabled_test(HandlerCase):
    handler = hash.unix_disabled
    known_correct_hashes = [('password', '!'), (UPASS_TABLE, '*')]
    known_unidentified_hashes = ['$1$xxx', 'abc', './az', '{SHA}xxx']

    def test_76_hash_border(self):
        self.accepts_all_hashes = True
        super(unix_disabled_test, self).test_76_hash_border()

    def test_90_special(self):
        """test marker option & special behavior"""
        warnings.filterwarnings('ignore', 'passing settings to .*.hash\\(\\) is deprecated')
        handler = self.handler
        self.assertEqual(handler.genhash('stub', '!asd'), '!asd')
        self.assertEqual(handler.genhash('stub', ''), handler.default_marker)
        self.assertEqual(handler.hash('stub'), handler.default_marker)
        self.assertEqual(handler.using().default_marker, handler.default_marker)
        self.assertEqual(handler.genhash('stub', '', marker='*xxx'), '*xxx')
        self.assertEqual(handler.hash('stub', marker='*xxx'), '*xxx')
        self.assertEqual(handler.using(marker='*xxx').hash('stub'), '*xxx')
        self.assertRaises(ValueError, handler.genhash, 'stub', '', marker='abc')
        self.assertRaises(ValueError, handler.hash, 'stub', marker='abc')
        self.assertRaises(ValueError, handler.using, marker='abc')