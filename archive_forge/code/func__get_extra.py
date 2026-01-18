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
def _get_extra(self):
    """Return extra settings of this commit."""
    warnings.warn('Commit.extra is deprecated. Use Commit._extra instead.', DeprecationWarning, stacklevel=2)
    return self._extra