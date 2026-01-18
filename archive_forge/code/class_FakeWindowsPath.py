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
class FakeWindowsPath(filepath.FilePath[AnyStr]):
    """
    A test version of FilePath which overrides listdir to raise L{WindowsError}.
    """

    def listdir(self) -> List[AnyStr]:
        """
        @raise WindowsError: always.
        """
        raise OSError(None, "A directory's validness was called into question", self.path, ERROR_DIRECTORY)