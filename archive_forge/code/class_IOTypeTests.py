import codecs
import io
import sys
import traceback
from unittest import skipIf
from twisted.python.compat import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SynchronousTestCase, TestCase
class IOTypeTests(SynchronousTestCase):
    """
    Test cases for determining a file-like object's type.
    """

    def test_3StringIO(self):
        """
        An L{io.StringIO} accepts and returns text.
        """
        self.assertEqual(ioType(io.StringIO()), str)

    def test_3BytesIO(self):
        """
        An L{io.BytesIO} accepts and returns bytes.
        """
        self.assertEqual(ioType(io.BytesIO()), bytes)

    def test_3openTextMode(self):
        """
        A file opened via 'io.open' in text mode accepts and returns text.
        """
        with open(self.mktemp(), 'w') as f:
            self.assertEqual(ioType(f), str)

    def test_3openBinaryMode(self):
        """
        A file opened via 'io.open' in binary mode accepts and returns bytes.
        """
        with open(self.mktemp(), 'wb') as f:
            self.assertEqual(ioType(f), bytes)

    def test_codecsOpenBytes(self):
        """
        The L{codecs} module, oddly, returns a file-like object which returns
        bytes when not passed an 'encoding' argument.
        """
        with codecs.open(self.mktemp(), 'wb') as f:
            self.assertEqual(ioType(f), bytes)

    def test_codecsOpenText(self):
        """
        When passed an encoding, however, the L{codecs} module returns unicode.
        """
        with codecs.open(self.mktemp(), 'wb', encoding='utf-8') as f:
            self.assertEqual(ioType(f), str)

    def test_defaultToText(self):
        """
        When passed an object about which no sensible decision can be made, err
        on the side of unicode.
        """
        self.assertEqual(ioType(object()), str)