import sys
from typing import List, Optional
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._io import LoggingFile
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
@implementer(ILogObserver)
class TestLoggingFile(LoggingFile):
    """
    L{LoggingFile} that is also an observer which captures events and messages.
    """

    def __init__(self, logger: Logger, level: NamedConstant=LogLevel.info, encoding: Optional[str]=None) -> None:
        super().__init__(logger=logger, level=level, encoding=encoding)
        self.events: List[LogEvent] = []
        self.messages: List[str] = []

    def __call__(self, event: LogEvent) -> None:
        self.events.append(event)
        if 'log_io' in event:
            self.messages.append(event['log_io'])