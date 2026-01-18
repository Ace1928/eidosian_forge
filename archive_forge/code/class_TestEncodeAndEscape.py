from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
class TestEncodeAndEscape(TestCase):
    """Whitebox testing of the _encode_and_escape function."""

    def setUp(self):
        super().setUp()
        breezy.bzr.xml_serializer._clear_cache()
        self.addCleanup(breezy.bzr.xml_serializer._clear_cache)

    def test_simple_ascii(self):
        val = breezy.bzr.xml_serializer.encode_and_escape('foo bar')
        self.assertEqual(b'foo bar', val)
        val2 = breezy.bzr.xml_serializer.encode_and_escape('foo bar')
        self.assertIs(val2, val)

    def test_ascii_with_xml(self):
        self.assertEqual(b'&amp;&apos;&quot;&lt;&gt;', breezy.bzr.xml_serializer.encode_and_escape('&\'"<>'))

    def test_utf8_with_xml(self):
        utf8_str = b'\xc2\xb5\xc3\xa5&\xd8\xac'
        self.assertEqual(b'&#181;&#229;&amp;&#1580;', breezy.bzr.xml_serializer.encode_and_escape(utf8_str))

    def test_unicode(self):
        uni_str = 'µå&ج'
        self.assertEqual(b'&#181;&#229;&amp;&#1580;', breezy.bzr.xml_serializer.encode_and_escape(uni_str))