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
def _parse_object(self, map) -> None:
    """Parse a new style object, setting self._text."""
    byte = ord(map[0:1])
    used = 1
    while byte & 128 != 0:
        byte = ord(map[used:used + 1])
        used += 1
    raw = map[used:]
    self.set_raw_string(_decompress(raw))