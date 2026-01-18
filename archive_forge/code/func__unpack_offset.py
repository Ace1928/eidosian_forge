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
def _unpack_offset(self, i):
    offset = self._pack_offset_table_offset + i * 4
    offset = unpack_from('>L', self._contents, offset)[0]
    if offset & 2 ** 31:
        offset = self._pack_offset_largetable_offset + (offset & 2 ** 31 - 1) * 8
        offset = unpack_from('>Q', self._contents, offset)[0]
    return offset