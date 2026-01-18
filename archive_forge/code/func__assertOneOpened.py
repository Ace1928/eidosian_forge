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
def _assertOneOpened(self, fp: TrackingFilePath[AnyStr], extension: str) -> None:
    """
        Assert that the L{TrackingFilePath} C{fp} was used to open one sibling
        with the given extension.

        @param fp: A L{TrackingFilePath} which should have been used to open
            file at a sibling path.
        @type fp: L{TrackingFilePath}

        @param extension: The extension the sibling path is expected to have
            had.
        @type extension: L{str}

        @raise: C{self.failureException} is raised if the extension of the
            opened file is incorrect or if not exactly one file was opened
            using C{fp}.
        """
    opened = fp.openedPaths()
    self.assertEqual(len(opened), 1, 'expected exactly one opened file')
    self.assertTrue(opened[0].asTextMode().basename().endswith(extension), '{!r} does not end with {!r} extension'.format(opened[0].basename(), extension))