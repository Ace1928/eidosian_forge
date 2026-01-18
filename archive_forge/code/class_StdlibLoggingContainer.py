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
class StdlibLoggingContainer:
    """
    Continer for a test configuration of stdlib logging objects.
    """

    def __init__(self) -> None:
        self.rootLogger = getLogger('')
        self.originalLevel = self.rootLogger.getEffectiveLevel()
        self.rootLogger.setLevel(py_logging.DEBUG)
        self.bufferedHandler = BufferedHandler()
        self.rootLogger.addHandler(self.bufferedHandler)
        self.streamHandler, self.output = handlerAndBytesIO()
        self.rootLogger.addHandler(self.streamHandler)

    def close(self) -> None:
        """
        Close the logger.
        """
        self.rootLogger.setLevel(self.originalLevel)
        self.rootLogger.removeHandler(self.bufferedHandler)
        self.rootLogger.removeHandler(self.streamHandler)
        self.streamHandler.close()
        self.output.close()

    def outputAsText(self) -> str:
        """
        Get the output to the underlying stream as text.

        @return: the output text
        """
        return self.output.getvalue().decode('utf-8')