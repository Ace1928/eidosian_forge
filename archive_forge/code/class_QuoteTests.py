import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class QuoteTests(TestCase):

    def test_quote(self):
        self.assertEqual('abc%20def', urlutils.quote('abc def'))
        self.assertEqual('abc%2Fdef', urlutils.quote('abc/def', safe=''))
        self.assertEqual('abc/def', urlutils.quote('abc/def', safe='/'))

    def test_quote_tildes(self):
        if sys.version_info[:2] >= (3, 7):
            self.assertEqual('~foo', urlutils.quote('~foo'))
        else:
            self.assertEqual('%7Efoo', urlutils.quote('~foo'))
        self.assertEqual('~foo', urlutils.quote('~foo', safe='/~'))

    def test_unquote(self):
        self.assertEqual('%', urlutils.unquote('%25'))
        self.assertEqual('å', urlutils.unquote('%C3%A5'))
        self.assertEqual('å', urlutils.unquote('å'))

    def test_unquote_to_bytes(self):
        self.assertEqual(b'%', urlutils.unquote_to_bytes('%25'))
        self.assertEqual(b'\xc3\xa5', urlutils.unquote_to_bytes('%C3%A5'))