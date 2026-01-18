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
def _read_fan_out_table(self, start_offset: int):
    ret = []
    for i in range(256):
        fanout_entry = self._contents[start_offset + i * 4:start_offset + (i + 1) * 4]
        ret.append(struct.unpack('>L', fanout_entry)[0])
    return ret