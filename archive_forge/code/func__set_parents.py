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
def _set_parents(self, value):
    """Set a list of parents of this commit."""
    self._needs_serialization = True
    self._parents = value