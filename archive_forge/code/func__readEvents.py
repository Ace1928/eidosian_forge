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