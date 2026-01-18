from __future__ import annotations
import calendar
import logging
import os
import sys
import time
import warnings
from io import IOBase, StringIO
from typing import Callable, List
from zope.interface import implementer
from typing_extensions import Protocol
from twisted.logger import (
from twisted.logger.test.test_stdlib import handlerAndBytesIO
from twisted.python import failure, log
from twisted.python.log import LogPublisher
from twisted.trial import unittest
class FileObserverTests(LogPublisherTestCaseMixin, unittest.SynchronousTestCase):
    """
    Tests for L{log.FileObserver}.
    """
    ERROR_INVALID_FORMAT = 'Invalid format string'
    ERROR_UNFORMATTABLE_OBJECT = 'UNFORMATTABLE OBJECT'
    ERROR_FORMAT = 'Invalid format string or unformattable object in log message'
    ERROR_PATHOLOGICAL = 'PATHOLOGICAL ERROR'
    ERROR_NO_FORMAT = 'Unable to format event'
    ERROR_UNFORMATTABLE_SYSTEM = '[UNFORMATTABLE]'
    ERROR_MESSAGE_LOST = 'MESSAGE LOST: unformattable object logged'

    def _getTimezoneOffsetTest(self, tzname: str, daylightOffset: int, standardOffset: int) -> None:
        """
        Verify that L{getTimezoneOffset} produces the expected offset for a
        certain timezone both when daylight saving time is in effect and when
        it is not.

        @param tzname: The name of a timezone to exercise.
        @type tzname: L{bytes}

        @param daylightOffset: The number of seconds west of UTC the timezone
            should be when daylight saving time is in effect.
        @type daylightOffset: L{int}

        @param standardOffset: The number of seconds west of UTC the timezone
            should be when daylight saving time is not in effect.
        @type standardOffset: L{int}
        """
        if getattr(time, 'tzset', None) is None:
            raise unittest.SkipTest('Platform cannot change timezone, cannot verify correct offsets in well-known timezones.')
        originalTimezone = os.environ.get('TZ', None)
        try:
            os.environ['TZ'] = tzname
            time.tzset()
            localStandardTuple = (2007, 1, 31, 0, 0, 0, 2, 31, 0)
            standard = time.mktime(localStandardTuple)
            localDaylightTuple = (2006, 6, 30, 0, 0, 0, 4, 181, 1)
            try:
                daylight = time.mktime(localDaylightTuple)
            except OverflowError:
                if daylightOffset == standardOffset:
                    daylight = standard
                else:
                    raise
            self.assertEqual((self.flo.getTimezoneOffset(daylight), self.flo.getTimezoneOffset(standard)), (daylightOffset, standardOffset))
        finally:
            if originalTimezone is None:
                del os.environ['TZ']
            else:
                os.environ['TZ'] = originalTimezone
            time.tzset()

    def test_getTimezoneOffsetWestOfUTC(self) -> None:
        """
        Attempt to verify that L{FileLogObserver.getTimezoneOffset} returns
        correct values for the current C{TZ} environment setting for at least
        some cases.  This test method exercises a timezone that is west of UTC
        (and should produce positive results).
        """
        self._getTimezoneOffsetTest('America/New_York', 14400, 18000)

    def test_getTimezoneOffsetEastOfUTC(self) -> None:
        """
        Attempt to verify that L{FileLogObserver.getTimezoneOffset} returns
        correct values for the current C{TZ} environment setting for at least
        some cases.  This test method exercises a timezone that is east of UTC
        (and should produce negative results).
        """
        self._getTimezoneOffsetTest('Europe/Berlin', -7200, -3600)

    def test_getTimezoneOffsetWithoutDaylightSavingTime(self) -> None:
        """
        Attempt to verify that L{FileLogObserver.getTimezoneOffset} returns
        correct values for the current C{TZ} environment setting for at least
        some cases.  This test method exercises a timezone that does not use
        daylight saving time at all (so both summer and winter time test values
        should have the same offset).
        """
        self._getTimezoneOffsetTest('Africa/Johannesburg', -7200, -7200)

    def test_timeFormatting(self) -> None:
        """
        Test the method of L{FileLogObserver} which turns a timestamp into a
        human-readable string.
        """
        when = calendar.timegm((2001, 2, 3, 4, 5, 6, 7, 8, 0))
        self.flo.getTimezoneOffset = lambda when: 18000
        self.assertEqual(self.flo.formatTime(when), '2001-02-02 23:05:06-0500')
        self.flo.getTimezoneOffset = lambda when: -3600
        self.assertEqual(self.flo.formatTime(when), '2001-02-03 05:05:06+0100')
        self.flo.getTimezoneOffset = lambda when: -39600
        self.assertEqual(self.flo.formatTime(when), '2001-02-03 15:05:06+1100')
        self.flo.getTimezoneOffset = lambda when: 5400
        self.assertEqual(self.flo.formatTime(when), '2001-02-03 02:35:06-0130')
        self.flo.getTimezoneOffset = lambda when: -5400
        self.assertEqual(self.flo.formatTime(when), '2001-02-03 05:35:06+0130')
        self.flo.getTimezoneOffset = lambda when: 1800
        self.assertEqual(self.flo.formatTime(when), '2001-02-03 03:35:06-0030')
        self.flo.getTimezoneOffset = lambda when: -1800
        self.assertEqual(self.flo.formatTime(when), '2001-02-03 04:35:06+0030')
        self.flo.timeFormat = '%Y %m'
        self.assertEqual(self.flo.formatTime(when), '2001 02')

    def test_microsecondTimestampFormatting(self) -> None:
        """
        L{FileLogObserver.formatTime} supports a value of C{timeFormat} which
        includes C{"%f"}, a L{datetime}-only format specifier for microseconds.
        """
        self.flo.timeFormat = '%f'
        self.assertEqual('600000', self.flo.formatTime(112345.6))

    def test_loggingAnObjectWithBroken__str__(self) -> None:
        self.lp.msg(EvilStr())
        self.assertEqual(len(self.out), 1)
        self.assertNotIn(self.ERROR_UNFORMATTABLE_OBJECT, self.out[0])

    def test_formattingAnObjectWithBroken__str__(self) -> None:
        self.lp.msg(format='%(blat)s', blat=EvilStr())
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_INVALID_FORMAT, self.out[0])

    def test_brokenSystem__str__(self) -> None:
        self.lp.msg('huh', system=EvilStr())
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_FORMAT, self.out[0])

    def test_formattingAnObjectWithBroken__repr__Indirect(self) -> None:
        self.lp.msg(format='%(blat)s', blat=[EvilRepr()])
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_UNFORMATTABLE_OBJECT, self.out[0])

    def test_systemWithBroker__repr__Indirect(self) -> None:
        self.lp.msg('huh', system=[EvilRepr()])
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_UNFORMATTABLE_OBJECT, self.out[0])

    def test_simpleBrokenFormat(self) -> None:
        self.lp.msg(format='hooj %s %s', blat=1)
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_INVALID_FORMAT, self.out[0])

    def test_ridiculousFormat(self) -> None:
        self.lp.msg(format=42, blat=1)
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_INVALID_FORMAT, self.out[0])

    def test_evilFormat__repr__And__str__(self) -> None:
        self.lp.msg(format=EvilReprStr(), blat=1)
        self.assertEqual(len(self.out), 1)
        self.assertIn(self.ERROR_PATHOLOGICAL, self.out[0])

    def test_strangeEventDict(self) -> None:
        """
        This kind of eventDict used to fail silently, so test it does.
        """
        self.lp.msg(message='', isError=False)
        self.assertEqual(len(self.out), 0)

    def _startLoggingCleanup(self) -> None:
        """
        Cleanup after a startLogging() call that mutates the hell out of some
        global state.
        """
        self.addCleanup(log.theLogPublisher._stopLogging)
        self.addCleanup(setattr, sys, 'stdout', sys.stdout)
        self.addCleanup(setattr, sys, 'stderr', sys.stderr)

    def test_printToStderrSetsIsError(self) -> None:
        """
        startLogging()'s overridden sys.stderr should consider everything
        written to it an error.
        """
        self._startLoggingCleanup()
        fakeFile = StringIO()
        log.startLogging(fakeFile)

        def observe(event: log.EventDict) -> None:
            observed.append(event)
        observed: list[log.EventDict] = []
        log.addObserver(observe)
        print('Hello, world.', file=sys.stderr)
        self.assertEqual(observed[0]['isError'], 1)

    def test_startLogging(self) -> None:
        """
        startLogging() installs FileLogObserver and overrides sys.stdout and
        sys.stderr.
        """
        origStdout, origStderr = (sys.stdout, sys.stderr)
        self._startLoggingCleanup()
        fakeFile = StringIO()
        observer = log.startLogging(fakeFile)
        self.addCleanup(observer.stop)
        log.msg('Hello!')
        self.assertIn('Hello!', fakeFile.getvalue())
        self.assertIsInstance(sys.stdout, LoggingFile)
        self.assertEqual(sys.stdout.level, NewLogLevel.info)
        encoding = getattr(origStdout, 'encoding', None)
        if not encoding:
            encoding = sys.getdefaultencoding()
        self.assertEqual(sys.stdout.encoding.upper(), encoding.upper())
        self.assertIsInstance(sys.stderr, LoggingFile)
        self.assertEqual(sys.stderr.level, NewLogLevel.error)
        encoding = getattr(origStderr, 'encoding', None)
        if not encoding:
            encoding = sys.getdefaultencoding()
        self.assertEqual(sys.stderr.encoding.upper(), encoding.upper())

    def test_startLoggingTwice(self) -> None:
        """
        There are some obscure error conditions that can occur when logging is
        started twice. See http://twistedmatrix.com/trac/ticket/3289 for more
        information.
        """
        self._startLoggingCleanup()
        sys.stdout = StringIO()

        def showError(eventDict: log.EventDict) -> None:
            if eventDict['isError']:
                sys.__stdout__.write(eventDict['failure'].getTraceback())
        log.addObserver(showError)
        self.addCleanup(log.removeObserver, showError)
        observer = log.startLogging(sys.stdout)
        self.addCleanup(observer.stop)
        self.assertIsInstance(sys.stdout, LoggingFile)
        fakeStdout = sys.stdout
        observer = log.startLogging(sys.stdout)
        self.assertIs(sys.stdout, fakeStdout)

    def test_startLoggingOverridesWarning(self) -> None:
        """
        startLogging() overrides global C{warnings.showwarning} such that
        warnings go to Twisted log observers.
        """
        self._startLoggingCleanup()
        newPublisher = NewLogPublisher()

        class SysModule:
            stdout = object()
            stderr = object()
        tempLogPublisher = LogPublisher(newPublisher, newPublisher, logBeginner=LogBeginner(newPublisher, StringIO(), SysModule, warnings))
        self.patch(log, 'theLogPublisher', tempLogPublisher)
        log._oldshowwarning = None
        fakeFile = StringIO()
        evt = {'pre-start': 'event'}
        received = []

        @implementer(ILogObserver)
        class PreStartObserver:

            def __call__(self, eventDict: log.EventDict) -> None:
                if 'pre-start' in eventDict.keys():
                    received.append(eventDict)
        newPublisher(evt)
        newPublisher.addObserver(PreStartObserver())
        log.startLogging(fakeFile, setStdout=False)
        self.addCleanup(tempLogPublisher._stopLogging)
        self.assertEqual(received, [])
        warnings.warn('hello!')
        output = fakeFile.getvalue()
        self.assertIn('UserWarning: hello!', output)

    def test_emitPrefix(self) -> None:
        """
        FileLogObserver.emit() will add a timestamp and system prefix to its
        file output.
        """
        output = StringIO()
        flo = log.FileLogObserver(output)
        events = []

        def observer(event: log.EventDict) -> None:
            events.append(event)
            flo.emit(event)
        publisher = log.LogPublisher()
        publisher.addObserver(observer)
        publisher.msg('Hello!')
        self.assertEqual(len(events), 1)
        event = events[0]
        result = output.getvalue()
        prefix = '{time} [{system}] '.format(time=flo.formatTime(event['time']), system=event['system'])
        self.assertTrue(result.startswith(prefix), f'{result!r} does not start with {prefix!r}')

    def test_emitNewline(self) -> None:
        """
        FileLogObserver.emit() will append a newline to its file output.
        """
        output = StringIO()
        flo = log.FileLogObserver(output)
        publisher = log.LogPublisher()
        publisher.addObserver(flo.emit)
        publisher.msg('Hello!')
        result = output.getvalue()
        suffix = 'Hello!\n'
        self.assertTrue(result.endswith(suffix), f'{result!r} does not end with {suffix!r}')