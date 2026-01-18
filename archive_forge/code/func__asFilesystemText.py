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
def _asFilesystemText(path: Union[bytes, str], encoding: Optional[str]=None) -> str:
    """
    Return C{path} as a string of L{unicode} suitable for use on this system's
    filesystem.

    @param path: The path to be made suitable.
    @type path: L{bytes} or L{unicode}

    @param encoding: The encoding to use if coercing to L{unicode}. If none
        is given, L{sys.getfilesystemencoding} is used.

    @return: L{unicode}
    """
    if isinstance(path, str):
        return path
    else:
        if encoding is None:
            encoding = sys.getfilesystemencoding()
        return path.decode(encoding, errors='surrogateescape')