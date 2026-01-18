from io import BytesIO, StringIO
from typing import IO, Any, List, Optional, Sequence, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._flatten import extractField
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._json import (
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
class LogFileReaderTests(TestCase):
    """
    Tests for L{eventsFromJSONLogFile}.
    """

    def setUp(self) -> None:
        self.errorEvents: List[LogEvent] = []

        @implementer(ILogObserver)
        def observer(event: LogEvent) -> None:
            if event['log_namespace'] == jsonLog.namespace and 'record' in event:
                self.errorEvents.append(event)
        self.logObserver = observer
        globalLogPublisher.addObserver(observer)

    def tearDown(self) -> None:
        globalLogPublisher.removeObserver(self.logObserver)

    def _readEvents(self, inFile: IO[Any], recordSeparator: Optional[str]=None, bufferSize: int=4096) -> None:
        """
        Test that L{eventsFromJSONLogFile} reads two pre-defined events from a
        file: C{{"x": 1}} and C{{"y": 2}}.

        @param inFile: C{inFile} argument to L{eventsFromJSONLogFile}
        @param recordSeparator: C{recordSeparator} argument to
            L{eventsFromJSONLogFile}
        @param bufferSize: C{bufferSize} argument to L{eventsFromJSONLogFile}
        """
        events = iter(eventsFromJSONLogFile(inFile, recordSeparator, bufferSize))
        self.assertEqual(next(events), {'x': 1})
        self.assertEqual(next(events), {'y': 2})
        self.assertRaises(StopIteration, next, events)

    def test_readEventsAutoWithRecordSeparator(self) -> None:
        """
        L{eventsFromJSONLogFile} reads events from a file and automatically
        detects use of C{"\\x1e"} as the record separator.
        """
        with StringIO('\x1e{"x": 1}\n\x1e{"y": 2}\n') as fileHandle:
            self._readEvents(fileHandle)
            self.assertEqual(len(self.errorEvents), 0)

    def test_readEventsAutoEmptyRecordSeparator(self) -> None:
        """
        L{eventsFromJSONLogFile} reads events from a file and automatically
        detects use of C{""} as the record separator.
        """
        with StringIO('{"x": 1}\n{"y": 2}\n') as fileHandle:
            self._readEvents(fileHandle)
            self.assertEqual(len(self.errorEvents), 0)

    def test_readEventsExplicitRecordSeparator(self) -> None:
        """
        L{eventsFromJSONLogFile} reads events from a file and is told to use
        a specific record separator.
        """
        with StringIO('\x08{"x": 1}\n\x08{"y": 2}\n') as fileHandle:
            self._readEvents(fileHandle, recordSeparator='\x08')
            self.assertEqual(len(self.errorEvents), 0)

    def test_readEventsPartialBuffer(self) -> None:
        """
        L{eventsFromJSONLogFile} handles buffering a partial event.
        """
        with StringIO('\x1e{"x": 1}\n\x1e{"y": 2}\n') as fileHandle:
            self._readEvents(fileHandle, bufferSize=1)
            self.assertEqual(len(self.errorEvents), 0)

    def test_readTruncated(self) -> None:
        """
        If the JSON text for a record is truncated, skip it.
        """
        with StringIO('\x1e{"x": 1\x1e{"y": 2}\n') as fileHandle:
            events = iter(eventsFromJSONLogFile(fileHandle))
            self.assertEqual(next(events), {'y': 2})
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 1)
            self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to read truncated JSON record: {record!r}')
            self.assertEqual(self.errorEvents[0]['record'], b'{"x": 1')

    def test_readUnicode(self) -> None:
        """
        If the file being read from vends L{str}, strings decode from JSON
        as-is.
        """
        with StringIO('\x1e{"currency": "€"}\n') as fileHandle:
            events = iter(eventsFromJSONLogFile(fileHandle))
            self.assertEqual(next(events), {'currency': '€'})
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 0)

    def test_readUTF8Bytes(self) -> None:
        """
        If the file being read from vends L{bytes}, strings decode from JSON as
        UTF-8.
        """
        with BytesIO(b'\x1e{"currency": "\xe2\x82\xac"}\n') as fileHandle:
            events = iter(eventsFromJSONLogFile(fileHandle))
            self.assertEqual(next(events), {'currency': '€'})
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 0)

    def test_readTruncatedUTF8Bytes(self) -> None:
        """
        If the JSON text for a record is truncated in the middle of a two-byte
        Unicode codepoint, we don't want to see a codec exception and the
        stream is read properly when the additional data arrives.
        """
        with BytesIO(b'\x1e{"x": "\xe2\x82\xac"}\n') as fileHandle:
            events = iter(eventsFromJSONLogFile(fileHandle, bufferSize=8))
            self.assertEqual(next(events), {'x': '€'})
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 0)

    def test_readInvalidUTF8Bytes(self) -> None:
        """
        If the JSON text for a record contains invalid UTF-8 text, ignore that
        record.
        """
        with BytesIO(b'\x1e{"x": "\xe2\xac"}\n\x1e{"y": 2}\n') as fileHandle:
            events = iter(eventsFromJSONLogFile(fileHandle))
            self.assertEqual(next(events), {'y': 2})
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 1)
            self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to decode UTF-8 for JSON record: {record!r}')
            self.assertEqual(self.errorEvents[0]['record'], b'{"x": "\xe2\xac"}\n')

    def test_readInvalidJSON(self) -> None:
        """
        If the JSON text for a record is invalid, skip it.
        """
        with StringIO('\x1e{"x": }\n\x1e{"y": 2}\n') as fileHandle:
            events = iter(eventsFromJSONLogFile(fileHandle))
            self.assertEqual(next(events), {'y': 2})
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 1)
            self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to read JSON record: {record!r}')
            self.assertEqual(self.errorEvents[0]['record'], b'{"x": }\n')

    def test_readUnseparated(self) -> None:
        """
        Multiple events without a record separator are skipped.
        """
        with StringIO('\x1e{"x": 1}\n{"y": 2}\n') as fileHandle:
            events = eventsFromJSONLogFile(fileHandle)
            self.assertRaises(StopIteration, next, events)
            self.assertEqual(len(self.errorEvents), 1)
            self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to read JSON record: {record!r}')
            self.assertEqual(self.errorEvents[0]['record'], b'{"x": 1}\n{"y": 2}\n')

    def test_roundTrip(self) -> None:
        """
        Data written by a L{FileLogObserver} returned by L{jsonFileLogObserver}
        and read by L{eventsFromJSONLogFile} is reconstructed properly.
        """
        event = dict(x=1)
        with StringIO() as fileHandle:
            observer = jsonFileLogObserver(fileHandle)
            observer(event)
            fileHandle.seek(0)
            events = eventsFromJSONLogFile(fileHandle)
            self.assertEqual(tuple(events), (event,))
            self.assertEqual(len(self.errorEvents), 0)