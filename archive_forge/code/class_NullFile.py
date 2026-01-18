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
class NullFile:
    """
    A file-like object that discards everything.
    """
    softspace = 0

    def read(self):
        """
        Do nothing.
        """

    def write(self, bytes):
        """
        Do nothing.

        @param bytes: data
        @type bytes: L{bytes}
        """

    def flush(self):
        """
        Do nothing.
        """

    def close(self):
        """
        Do nothing.
        """