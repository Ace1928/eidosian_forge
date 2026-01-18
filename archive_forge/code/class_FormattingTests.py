from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
class FormattingTests(unittest.TestCase):
    """
    Tests for basic event formatting functions.
    """

    def format(self, logFormat: AnyStr, **event: object) -> str:
        """
        Create a Twisted log event dictionary from C{event} with the given
        C{logFormat} format string, format it with L{formatEvent}, ensure that
        its type is L{str}, and return its result.
        """
        event['log_format'] = logFormat
        result = formatEvent(event)
        self.assertIs(type(result), str)
        return result

    def test_formatEvent(self) -> None:
        """
        L{formatEvent} will format an event according to several rules:

            - A string with no formatting instructions will be passed straight
              through.

            - PEP 3101 strings will be formatted using the keys and values of
              the event as named fields.

            - PEP 3101 keys ending with C{()} will be treated as instructions
              to call that key (which ought to be a callable) before
              formatting.

        L{formatEvent} will always return L{str}, and if given bytes, will
        always treat its format string as UTF-8 encoded.
        """
        self.assertEqual('', self.format(b''))
        self.assertEqual('', self.format(''))
        self.assertEqual('abc', self.format('{x}', x='abc'))
        self.assertEqual('no, yes.', self.format('{not_called}, {called()}.', not_called='no', called=lambda: 'yes'))
        self.assertEqual('Sánchez', self.format(b'S\xc3\xa1nchez'))
        self.assertIn('Unable to format event', self.format(b'S\xe1nchez'))
        maybeResult = self.format(b'S{a!s}nchez', a=b'\xe1')
        self.assertIn("Sb'\\xe1'nchez", maybeResult)
        xe1 = str(repr(b'\xe1'))
        self.assertIn('S' + xe1 + 'nchez', self.format(b'S{a!r}nchez', a=b'\xe1'))

    def test_formatMethod(self) -> None:
        """
        L{formatEvent} will format PEP 3101 keys containing C{.}s ending with
        C{()} as methods.
        """

        class World:

            def where(self) -> str:
                return 'world'
        self.assertEqual('hello world', self.format('hello {what.where()}', what=World()))

    def test_formatAttributeSubscript(self) -> None:
        """
        L{formatEvent} will format subscripts of attributes per PEP 3101.
        """

        class Example(object):
            config: Dict[str, str] = dict(foo='bar', baz='qux')
        self.assertEqual('bar qux', self.format('{example.config[foo]} {example.config[baz]}', example=Example()))

    def test_formatEventNoFormat(self) -> None:
        """
        Formatting an event with no format.
        """
        event = dict(foo=1, bar=2)
        result = formatEvent(event)
        self.assertEqual('', result)

    def test_formatEventWeirdFormat(self) -> None:
        """
        Formatting an event with a bogus format.
        """
        event = dict(log_format=object(), foo=1, bar=2)
        result = formatEvent(event)
        self.assertIn('Log format must be str', result)
        self.assertIn(repr(event), result)

    def test_formatUnformattableEvent(self) -> None:
        """
        Formatting an event that's just plain out to get us.
        """
        event = dict(log_format='{evil()}', evil=lambda: 1 / 0)
        result = formatEvent(event)
        self.assertIn('Unable to format event', result)
        self.assertIn(repr(event), result)

    def test_formatUnformattableEventWithUnformattableKey(self) -> None:
        """
        Formatting an unformattable event that has an unformattable key.
        """
        event: LogEvent = {'log_format': '{evil()}', 'evil': lambda: 1 / 0, cast(str, Unformattable()): 'gurk'}
        result = formatEvent(event)
        self.assertIn('MESSAGE LOST: unformattable object logged:', result)
        self.assertIn('Recoverable data:', result)
        self.assertIn('Exception during formatting:', result)

    def test_formatUnformattableEventWithUnformattableValue(self) -> None:
        """
        Formatting an unformattable event that has an unformattable value.
        """
        event = dict(log_format='{evil()}', evil=lambda: 1 / 0, gurk=Unformattable())
        result = formatEvent(event)
        self.assertIn('MESSAGE LOST: unformattable object logged:', result)
        self.assertIn('Recoverable data:', result)
        self.assertIn('Exception during formatting:', result)

    def test_formatUnformattableEventWithUnformattableErrorOMGWillItStop(self) -> None:
        """
        Formatting an unformattable event that has an unformattable value.
        """
        event = dict(log_format='{evil()}', evil=lambda: 1 / 0, recoverable='okay')
        result = formatUnformattableEvent(event, cast(BaseException, Unformattable()))
        self.assertIn('MESSAGE LOST: unformattable object logged:', result)
        self.assertIn(repr('recoverable') + ' = ' + repr('okay'), result)