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
class DefaultObserver(_GlobalStartStopObserver):
    """
    Default observer.

    Will ignore all non-error messages and send error messages to sys.stderr.
    Will be removed when startLogging() is called for the first time.
    """
    stderr = sys.stderr

    def emit(self, eventDict: EventDict) -> None:
        """
        Emit an event dict.

        @param eventDict: an event dict
        """
        if eventDict['isError']:
            text = textFromEventDict(eventDict)
            if text is not None:
                self.stderr.write(text)
            self.stderr.flush()