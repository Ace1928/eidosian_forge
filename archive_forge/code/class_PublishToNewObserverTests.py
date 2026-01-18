import logging as py_logging
from time import time
from typing import List, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python import context, log as legacyLog
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._format import formatEvent
from .._interfaces import ILogObserver, LogEvent
from .._legacy import LegacyLogObserverWrapper, publishToNewObserver
from .._levels import LogLevel
class PublishToNewObserverTests(unittest.TestCase):
    """
    Tests for L{publishToNewObserver}.
    """

    def setUp(self) -> None:
        self.events: List[LogEvent] = []
        self.observer = cast(ILogObserver, self.events.append)

    def legacyEvent(self, *message: str, **values: object) -> legacyLog.EventDict:
        """
        Return a basic old-style event as would be created by L{legacyLog.msg}.

        @param message: a message event value in the legacy event format
        @param values: additional event values in the legacy event format

        @return: a legacy event
        """
        event = (context.get(legacyLog.ILogContext) or {}).copy()
        event.update(values)
        event['message'] = message
        event['time'] = time()
        if 'isError' not in event:
            event['isError'] = 0
        return event

    def test_observed(self) -> None:
        """
        The observer is called exactly once.
        """
        publishToNewObserver(self.observer, self.legacyEvent(), legacyLog.textFromEventDict)
        self.assertEqual(len(self.events), 1)

    def test_time(self) -> None:
        """
        The old-style C{"time"} key is copied to the new-style C{"log_time"}
        key.
        """
        publishToNewObserver(self.observer, self.legacyEvent(), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_time'], self.events[0]['time'])

    def test_message(self) -> None:
        """
        A published old-style event should format as text in the same way as
        the given C{textFromEventDict} callable would format it.
        """

        def textFromEventDict(event: LogEvent) -> str:
            return ''.join(reversed(' '.join(event['message'])))
        event = self.legacyEvent('Hello,', 'world!')
        text = textFromEventDict(event)
        publishToNewObserver(self.observer, event, textFromEventDict)
        self.assertEqual(formatEvent(self.events[0]), text)

    def test_defaultLogLevel(self) -> None:
        """
        Published event should have log level of L{LogLevel.info}.
        """
        publishToNewObserver(self.observer, self.legacyEvent(), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_level'], LogLevel.info)

    def test_isError(self) -> None:
        """
        If C{"isError"} is set to C{1} (true) on the legacy event, the
        C{"log_level"} key should get set to L{LogLevel.critical}.
        """
        publishToNewObserver(self.observer, self.legacyEvent(isError=1), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_level'], LogLevel.critical)

    def test_stdlibLogLevel(self) -> None:
        """
        If the old-style C{"logLevel"} key is set to a standard library logging
        level, using a predefined (L{int}) constant, the new-style
        C{"log_level"} key should get set to the corresponding log level.
        """
        publishToNewObserver(self.observer, self.legacyEvent(logLevel=py_logging.WARNING), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_level'], LogLevel.warn)

    def test_stdlibLogLevelWithString(self) -> None:
        """
        If the old-style C{"logLevel"} key is set to a standard library logging
        level, using a string value, the new-style C{"log_level"} key should
        get set to the corresponding log level.
        """
        publishToNewObserver(self.observer, self.legacyEvent(logLevel='WARNING'), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_level'], LogLevel.warn)

    def test_stdlibLogLevelWithGarbage(self) -> None:
        """
        If the old-style C{"logLevel"} key is set to a standard library logging
        level, using an unknown value, the new-style C{"log_level"} key should
        not get set.
        """
        publishToNewObserver(self.observer, self.legacyEvent(logLevel='Foo!!!!!'), legacyLog.textFromEventDict)
        self.assertNotIn('log_level', self.events[0])

    def test_defaultNamespace(self) -> None:
        """
        Published event should have a namespace of C{"log_legacy"} to indicate
        that it was forwarded from legacy logging.
        """
        publishToNewObserver(self.observer, self.legacyEvent(), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_namespace'], 'log_legacy')

    def test_system(self) -> None:
        """
        The old-style C{"system"} key is copied to the new-style
        C{"log_system"} key.
        """
        publishToNewObserver(self.observer, self.legacyEvent(), legacyLog.textFromEventDict)
        self.assertEqual(self.events[0]['log_system'], self.events[0]['system'])