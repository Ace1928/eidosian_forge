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
def _asTextPath(self, encoding: Optional[str]=None) -> str:
    """
        Return the path of this L{FilePath} as text.

        @param encoding: The encoding to use if coercing to L{unicode}. If none
            is given, L{sys.getfilesystemencoding} is used.

        @return: L{unicode}
        """
    return _asFilesystemText(self.path, encoding=encoding)