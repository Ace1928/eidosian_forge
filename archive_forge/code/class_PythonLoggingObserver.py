import sys
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, BinaryIO, Dict, Optional, cast
from zope.interface import Interface
from twisted.logger import (
from twisted.logger._global import LogBeginner
from twisted.logger._legacy import publishToNewObserver as _publishNew
from twisted.python import context, failure, reflect, util
from twisted.python.threadable import synchronize
class PythonLoggingObserver(_GlobalStartStopObserver):
    """
    Output twisted messages to Python standard library L{logging} module.

    WARNING: specific logging configurations (example: network) can lead to
    a blocking system. Nothing is done here to prevent that, so be sure to not
    use this: code within Twisted, such as twisted.web, assumes that logging
    does not block.
    """

    def __init__(self, loggerName='twisted'):
        """
        @param loggerName: identifier used for getting logger.
        @type loggerName: C{str}
        """
        self._newObserver = NewSTDLibLogObserver(loggerName)

    def emit(self, eventDict: EventDict) -> None:
        """
        Receive a twisted log entry, format it and bridge it to python.

        By default the logging level used is info; log.err produces error
        level, and you can customize the level by using the C{logLevel} key::

            >>> log.msg('debugging', logLevel=logging.DEBUG)
        """
        if 'log_format' in eventDict:
            _publishNew(self._newObserver, eventDict, textFromEventDict)