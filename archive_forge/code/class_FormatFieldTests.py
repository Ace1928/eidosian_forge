from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
class FormatFieldTests(unittest.TestCase):
    """
    Tests for format field functions.
    """

    def test_formatWithCall(self) -> None:
        """
        L{formatWithCall} is an extended version of L{str.format} that
        will interpret a set of parentheses "C{()}" at the end of a format key
        to mean that the format key ought to be I{called} rather than
        stringified.
        """
        self.assertEqual(formatWithCall('Hello, {world}. {callme()}.', dict(world='earth', callme=lambda: 'maybe')), 'Hello, earth. maybe.')
        self.assertEqual(formatWithCall('Hello, {repr()!r}.', dict(repr=lambda: 'repr')), "Hello, 'repr'.")