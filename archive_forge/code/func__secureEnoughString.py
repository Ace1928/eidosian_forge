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
def _secureEnoughString(path: AnyStr) -> AnyStr:
    """
    Compute a string usable as a new, temporary filename.

    @param path: The path that the new temporary filename should be able to be
        concatenated with.

    @return: A pseudorandom, 16 byte string for use in secure filenames.
    @rtype: the type of C{path}
    """
    secureishString = armor(randomBytes(16))[:16]
    return _coerceToFilesystemEncoding(path, secureishString)