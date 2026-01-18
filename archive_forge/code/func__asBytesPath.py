from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
def _asBytesPath(self, encoding: Optional[str]=None) -> bytes:
    """
        Return the path of this L{FilePath} as bytes.

        @param encoding: The encoding to use if coercing to L{bytes}. If none is
            given, L{sys.getfilesystemencoding} is used.

        @return: L{bytes}
        """
    return _asFilesystemBytes(self.path, encoding=encoding)