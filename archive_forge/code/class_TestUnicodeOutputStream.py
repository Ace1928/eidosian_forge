import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
class TestUnicodeOutputStream(testtools.TestCase):
    """Test wrapping output streams so they work with arbitrary unicode"""
    uni = 'paɪθən'

    def setUp(self):
        super().setUp()
        if sys.platform == 'cli':
            self.skipTest("IronPython shouldn't wrap streams to do encoding")

    def test_no_encoding_becomes_ascii(self):
        """A stream with no encoding attribute gets ascii/replace strings"""
        sout = _FakeOutputStream()
        unicode_output_stream(sout).write(self.uni)
        self.assertEqual([_b('pa???n')], sout.writelog)

    def test_encoding_as_none_becomes_ascii(self):
        """A stream with encoding value of None gets ascii/replace strings"""
        sout = _FakeOutputStream()
        sout.encoding = None
        unicode_output_stream(sout).write(self.uni)
        self.assertEqual([_b('pa???n')], sout.writelog)

    def test_bogus_encoding_becomes_ascii(self):
        """A stream with a bogus encoding gets ascii/replace strings"""
        sout = _FakeOutputStream()
        sout.encoding = 'bogus'
        unicode_output_stream(sout).write(self.uni)
        self.assertEqual([_b('pa???n')], sout.writelog)

    def test_partial_encoding_replace(self):
        """A string which can be partly encoded correctly should be"""
        sout = _FakeOutputStream()
        sout.encoding = 'iso-8859-7'
        unicode_output_stream(sout).write(self.uni)
        self.assertEqual([_b('pa?è?n')], sout.writelog)

    def test_stringio(self):
        """A StringIO object should maybe get an ascii native str type"""
        sout = io.StringIO()
        soutwrapper = unicode_output_stream(sout)
        soutwrapper.write(self.uni)
        self.assertEqual(self.uni, sout.getvalue())

    def test_io_stringio(self):
        s = io.StringIO()
        self.assertEqual(s, unicode_output_stream(s))

    def test_io_bytesio(self):
        bytes_io = io.BytesIO()
        self.assertThat(bytes_io, Not(Is(unicode_output_stream(bytes_io))))
        unicode_output_stream(bytes_io).write('foo')

    def test_io_textwrapper(self):
        text_io = io.TextIOWrapper(io.BytesIO())
        self.assertThat(unicode_output_stream(text_io), Is(text_io))
        unicode_output_stream(text_io).write('foo')