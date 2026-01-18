import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class WriterTests(unittest.TestCase):

    def test_Integer(self):
        """
        Colorize an integer.
        """
        manhole.lastColorizedLine('1')

    def test_DoubleQuoteString(self):
        """
        Colorize an integer in double quotes.
        """
        manhole.lastColorizedLine('"1"')

    def test_SingleQuoteString(self):
        """
        Colorize an integer in single quotes.
        """
        manhole.lastColorizedLine("'1'")

    def test_TripleSingleQuotedString(self):
        """
        Colorize an integer in triple quotes.
        """
        manhole.lastColorizedLine("'''1'''")

    def test_TripleDoubleQuotedString(self):
        """
        Colorize an integer in triple and double quotes.
        """
        manhole.lastColorizedLine('"""1"""')

    def test_FunctionDefinition(self):
        """
        Colorize a function definition.
        """
        manhole.lastColorizedLine('def foo():')

    def test_ClassDefinition(self):
        """
        Colorize a class definition.
        """
        manhole.lastColorizedLine('class foo:')

    def test_unicode(self):
        """
        Colorize a Unicode string.
        """
        res = manhole.lastColorizedLine('и')
        self.assertTrue(isinstance(res, bytes))

    def test_bytes(self):
        """
        Colorize a UTF-8 byte string.
        """
        res = manhole.lastColorizedLine(b'\xd0\xb8')
        self.assertTrue(isinstance(res, bytes))

    def test_identicalOutput(self):
        """
        The output of UTF-8 bytestrings and Unicode strings are identical.
        """
        self.assertEqual(manhole.lastColorizedLine(b'\xd0\xb8'), manhole.lastColorizedLine('и'))