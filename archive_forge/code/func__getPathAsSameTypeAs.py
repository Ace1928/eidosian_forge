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
def _getPathAsSameTypeAs(self, pattern: OtherAnyStr) -> OtherAnyStr:
    """
        If C{pattern} is C{bytes}, return L{FilePath.path} as L{bytes}.
        Otherwise, return L{FilePath.path} as L{unicode}.

        @param pattern: The new element of the path that L{FilePath.path} may
            need to be coerced to match.
        """
    if isinstance(pattern, bytes):
        return self._asBytesPath()
    else:
        return self._asTextPath()