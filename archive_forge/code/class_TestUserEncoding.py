import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
class TestUserEncoding(TestCase):
    """Test detection of default user encoding."""

    def setUp(self):
        super().setUp()
        self.overrideAttr(osutils, '_cached_user_encoding', None)
        self.overrideAttr(locale, 'getpreferredencoding', self.get_encoding)
        self.overrideAttr(locale, 'CODESET', None)
        self.overrideAttr(sys, 'stderr', StringIOWithEncoding())

    def get_encoding(self, do_setlocale=True):
        return self._encoding

    def test_get_user_encoding(self):
        self._encoding = 'user_encoding'
        fake_codec.add('user_encoding')
        self.assertEqual('iso8859-1', osutils.get_user_encoding())
        self.assertEqual('', sys.stderr.getvalue())

    def test_user_cp0(self):
        self._encoding = 'cp0'
        self.assertEqual('ascii', osutils.get_user_encoding())
        self.assertEqual('', sys.stderr.getvalue())

    def test_user_cp_unknown(self):
        self._encoding = 'cp-unknown'
        self.assertEqual('ascii', osutils.get_user_encoding())
        self.assertEqual('brz: warning: unknown encoding cp-unknown. Continuing with ascii encoding.\n', sys.stderr.getvalue())

    def test_user_empty(self):
        """Running bzr from a vim script gives '' for a preferred locale"""
        self._encoding = ''
        self.assertEqual('ascii', osutils.get_user_encoding())
        self.assertEqual('', sys.stderr.getvalue())