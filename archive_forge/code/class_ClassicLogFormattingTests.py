from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
class ClassicLogFormattingTests(unittest.TestCase):
    """
    Tests for classic text log event formatting functions.
    """

    def test_formatTimeDefault(self) -> None:
        """
        Time is first field.  Default time stamp format is RFC 3339 and offset
        respects the timezone as set by the standard C{TZ} environment variable
        and L{tzset} API.
        """
        if tzset is None:
            raise SkipTest('Platform cannot change timezone; unable to verify offsets.')
        addTZCleanup(self)
        setTZ('UTC+00')
        t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
        event = dict(log_format='XYZZY', log_time=t)
        self.assertEqual(formatEventAsClassicLogText(event), '2013-09-24T11:40:47+0000 [-#-] XYZZY\n')

    def test_formatTimeCustom(self) -> None:
        """
        Time is first field.  Custom formatting function is an optional
        argument.
        """

        def formatTime(t: Optional[float]) -> str:
            return f'__{t}__'
        event = dict(log_format='XYZZY', log_time=12345)
        self.assertEqual(formatEventAsClassicLogText(event, formatTime=formatTime), '__12345__ [-#-] XYZZY\n')

    def test_formatNamespace(self) -> None:
        """
        Namespace is first part of second field.
        """
        event = dict(log_format='XYZZY', log_namespace='my.namespace')
        self.assertEqual(formatEventAsClassicLogText(event), '- [my.namespace#-] XYZZY\n')

    def test_formatLevel(self) -> None:
        """
        Level is second part of second field.
        """
        event = dict(log_format='XYZZY', log_level=LogLevel.warn)
        self.assertEqual(formatEventAsClassicLogText(event), '- [-#warn] XYZZY\n')

    def test_formatSystem(self) -> None:
        """
        System is second field.
        """
        event = dict(log_format='XYZZY', log_system='S.Y.S.T.E.M.')
        self.assertEqual(formatEventAsClassicLogText(event), '- [S.Y.S.T.E.M.] XYZZY\n')

    def test_formatSystemRulz(self) -> None:
        """
        System is not supplanted by namespace and level.
        """
        event = dict(log_format='XYZZY', log_namespace='my.namespace', log_level=LogLevel.warn, log_system='S.Y.S.T.E.M.')
        self.assertEqual(formatEventAsClassicLogText(event), '- [S.Y.S.T.E.M.] XYZZY\n')

    def test_formatSystemUnformattable(self) -> None:
        """
        System is not supplanted by namespace and level.
        """
        event = dict(log_format='XYZZY', log_system=Unformattable())
        self.assertEqual(formatEventAsClassicLogText(event), '- [UNFORMATTABLE] XYZZY\n')

    def test_formatFormat(self) -> None:
        """
        Formatted event is last field.
        """
        event = dict(log_format='id:{id}', id='123')
        self.assertEqual(formatEventAsClassicLogText(event), '- [-#-] id:123\n')

    def test_formatNoFormat(self) -> None:
        """
        No format string.
        """
        event = dict(id='123')
        self.assertIs(formatEventAsClassicLogText(event), None)

    def test_formatEmptyFormat(self) -> None:
        """
        Empty format string.
        """
        event = dict(log_format='', id='123')
        self.assertIs(formatEventAsClassicLogText(event), None)

    def test_formatFormatMultiLine(self) -> None:
        """
        If the formatted event has newlines, indent additional lines.
        """
        event = dict(log_format='XYZZY\nA hollow voice says:\n"Plugh"')
        self.assertEqual(formatEventAsClassicLogText(event), '- [-#-] XYZZY\n\tA hollow voice says:\n\t"Plugh"\n')