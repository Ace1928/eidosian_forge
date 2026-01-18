import binascii
import os
import posixpath
import stat
import warnings
import zlib
from collections import namedtuple
from hashlib import sha1
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
def S_ISGITLINK(m):
    """Check if a mode indicates a submodule.

    Args:
      m: Mode to check
    Returns: a ``boolean``
    """
    return stat.S_IFMT(m) == S_IFGITLINK