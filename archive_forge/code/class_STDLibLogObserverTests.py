from __future__ import annotations
import logging as py_logging
import sys
from inspect import getsourcefile
from io import BytesIO, TextIOWrapper
from logging import Formatter, LogRecord, StreamHandler, getLogger
from typing import List, Optional, Tuple
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.compat import currentframe
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._stdlib import STDLibLogObserver
class STDLibLogObserverTests(unittest.TestCase):
    """
    Tests for L{STDLibLogObserver}.
    """

    def test_interface(self) -> None:
        """
        L{STDLibLogObserver} is an L{ILogObserver}.
        """
        observer = STDLibLogObserver()
        try:
            verifyObject(ILogObserver, observer)
        except BrokenMethodImplementation as e:
            self.fail(e)

    def py_logger(self) -> StdlibLoggingContainer:
        """
        Create a logging object we can use to test with.

        @return: a stdlib-style logger
        """
        logger = StdlibLoggingContainer()
        self.addCleanup(logger.close)
        return logger

    def logEvent(self, *events: LogEvent) -> Tuple[List[LogRecord], str]:
        """
        Send one or more events to Python's logging module, and capture the
        emitted L{LogRecord}s and output stream as a string.

        @param events: events

        @return: a tuple: (records, output)
        """
        pl = self.py_logger()
        observer = STDLibLogObserver(stackDepth=STDLibLogObserver.defaultStackDepth + 1)
        for event in events:
            observer(event)
        return (pl.bufferedHandler.records, pl.outputAsText())

    def test_name(self) -> None:
        """
        Logger name.
        """
        records, output = self.logEvent({})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].name, 'twisted')

    def test_levels(self) -> None:
        """
        Log levels.
        """
        levelMapping = {None: py_logging.INFO, LogLevel.debug: py_logging.DEBUG, LogLevel.info: py_logging.INFO, LogLevel.warn: py_logging.WARNING, LogLevel.error: py_logging.ERROR, LogLevel.critical: py_logging.CRITICAL}
        events = []
        for level, pyLevel in levelMapping.items():
            event = {}
            if level is not None:
                event['log_level'] = level
            event['py_levelno'] = int(pyLevel)
            events.append(event)
        records, output = self.logEvent(*events)
        self.assertEqual(len(records), len(levelMapping))
        for i in range(len(records)):
            self.assertEqual(records[i].levelno, events[i]['py_levelno'])

    def test_callerInfo(self) -> None:
        """
        C{pathname}, C{lineno}, C{exc_info}, C{func} is set properly on
        records.
        """
        filename, logLine = nextLine()
        records, output = self.logEvent({})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].pathname, filename)
        self.assertEqual(records[0].lineno, logLine)
        self.assertIsNone(records[0].exc_info)

    def test_basicFormat(self) -> None:
        """
        Basic formattable event passes the format along correctly.
        """
        event = dict(log_format='Hello, {who}!', who='dude')
        records, output = self.logEvent(event)
        self.assertEqual(len(records), 1)
        self.assertEqual(str(records[0].msg), 'Hello, dude!')
        self.assertEqual(records[0].args, ())

    def test_basicFormatRendered(self) -> None:
        """
        Basic formattable event renders correctly.
        """
        event = dict(log_format='Hello, {who}!', who='dude')
        records, output = self.logEvent(event)
        self.assertEqual(len(records), 1)
        self.assertTrue(output.endswith(':Hello, dude!\n'), repr(output))

    def test_noFormat(self) -> None:
        """
        Event with no format.
        """
        records, output = self.logEvent({})
        self.assertEqual(len(records), 1)
        self.assertEqual(str(records[0].msg), '')

    def test_failure(self) -> None:
        """
        An event with a failure logs the failure details as well.
        """

        def failing_func() -> None:
            1 / 0
        try:
            failing_func()
        except ZeroDivisionError:
            failure = Failure()
        event = dict(log_format='Hi mom', who='me', log_failure=failure)
        records, output = self.logEvent(event)
        self.assertEqual(len(records), 1)
        self.assertIn('Hi mom', output)
        self.assertIn('in failing_func', output)
        self.assertIn('ZeroDivisionError', output)

    def test_cleanedFailure(self) -> None:
        """
        A cleaned Failure object has a fake traceback object; make sure that
        logging such a failure still results in the exception details being
        logged.
        """

        def failing_func() -> None:
            1 / 0
        try:
            failing_func()
        except ZeroDivisionError:
            failure = Failure()
            failure.cleanFailure()
        event = dict(log_format='Hi mom', who='me', log_failure=failure)
        records, output = self.logEvent(event)
        self.assertEqual(len(records), 1)
        self.assertIn('Hi mom', output)
        self.assertIn('in failing_func', output)
        self.assertIn('ZeroDivisionError', output)