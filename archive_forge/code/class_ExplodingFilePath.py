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
class ExplodingFilePath(filepath.FilePath[AnyStr]):
    """
    A specialized L{FilePath} which always returns an instance of
    L{ExplodingFile} from its C{open} method.

    @ivar fp: The L{ExplodingFile} instance most recently returned from the
        C{open} method.
    """
    fp: ExplodingFile

    def __init__(self, pathName: AnyStr, originalExploder: Optional[Union[ExplodingFilePath[str], ExplodingFilePath[bytes]]]=None) -> None:
        """
        Initialize an L{ExplodingFilePath} with a name and a reference to the

        @param pathName: The path name as passed to L{filepath.FilePath}.
        @type pathName: C{str}

        @param originalExploder: The L{ExplodingFilePath} to associate opened
        files with.
        @type originalExploder: L{ExplodingFilePath}
        """
        filepath.FilePath.__init__(self, pathName)
        if originalExploder is None:
            originalExploder = self
        self._originalExploder = originalExploder

    def open(self, mode: FileMode='r') -> IO[bytes]:
        """
        Create, save, and return a new C{ExplodingFile}.

        @param mode: Present for signature compatibility.  Ignored.

        @return: A new C{ExplodingFile}.
        """
        f = self._originalExploder.fp = ExplodingFile()
        return f

    def clonePath(self, name: OtherAnyStr, alwaysCreate: bool=False) -> filepath.FilePath[OtherAnyStr]:
        return ExplodingFilePath(name, self._originalExploder)