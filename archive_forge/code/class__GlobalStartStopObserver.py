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
class _GlobalStartStopObserver(ABC):
    """
    Mix-in for global log observers that can start and stop.
    """

    @abstractmethod
    def emit(self, eventDict: EventDict) -> None:
        """
        Emit the given log event.

        @param eventDict: a log event
        """

    def start(self) -> None:
        """
        Start observing log events.
        """
        addObserver(self.emit)

    def stop(self) -> None:
        """
        Stop observing log events.
        """
        removeObserver(self.emit)