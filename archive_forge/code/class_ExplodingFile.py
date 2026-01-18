from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
class ExplodingFile:
    """
    A C{file}-alike which raises exceptions from its I/O methods and keeps track
    of whether it has been closed.

    @ivar closed: A C{bool} which is C{False} until C{close} is called, then it
        is C{True}.
    """
    closed = False

    def read(self, n: int=0) -> NoReturn:
        """
        @raise IOError: Always raised.
        """
        raise OSError()

    def write(self, what: bytes) -> NoReturn:
        """
        @raise IOError: Always raised.
        """
        raise OSError()

    def close(self) -> None:
        """
        Mark the file as having been closed.
        """
        self.closed = True

    def __enter__(self) -> ExplodingFile:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()