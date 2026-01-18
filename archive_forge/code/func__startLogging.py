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
def _startLogging(self, other, setStdout):
    """
        Begin logging to the L{LogBeginner} associated with this
        L{LogPublisher}.

        @param other: the observer to log to.
        @type other: L{LogBeginner}

        @param setStdout: if true, send standard I/O to the observer as well.
        @type setStdout: L{bool}
        """
    wrapped = LegacyLogObserverWrapper(other)
    self._legacyObservers.append(wrapped)
    self._logBeginner.beginLoggingTo([wrapped], True, setStdout)