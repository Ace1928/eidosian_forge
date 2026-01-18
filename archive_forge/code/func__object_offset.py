import binascii
from collections import defaultdict, deque
from contextlib import suppress
from io import BytesIO, UnsupportedOperation
import os
import struct
import sys
from itertools import chain
from typing import (
import warnings
import zlib
from hashlib import sha1
from os import SEEK_CUR, SEEK_END
from struct import unpack_from
from .errors import ApplyDeltaError, ChecksumMismatch
from .file import GitFile
from .lru_cache import LRUSizeCache
from .objects import ObjectID, ShaFile, hex_to_sha, object_header, sha_to_hex
def _object_offset(self, sha: bytes) -> int:
    """See object_offset.

        Args:
          sha: A *binary* SHA string. (20 characters long)_
        """
    assert len(sha) == 20
    idx = ord(sha[:1])
    if idx == 0:
        start = 0
    else:
        start = self._fan_out_table[idx - 1]
    end = self._fan_out_table[idx]
    i = bisect_find_sha(start, end, sha, self._unpack_name)
    if i is None:
        raise KeyError(sha)
    return self._unpack_offset(i)