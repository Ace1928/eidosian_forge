from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
class LogComposedObject:
    """
    A regular object, with a logger attached.
    """
    log = TestLogger()

    def __init__(self, state: Optional[str]=None) -> None:
        self.state = state

    def __str__(self) -> str:
        return f'<LogComposedObject {self.state}>'