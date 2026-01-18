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
def _encode_copy_operation(start, length):
    scratch = bytearray([128])
    for i in range(4):
        if start & 255 << i * 8:
            scratch.append(start >> i * 8 & 255)
            scratch[0] |= 1 << i
    for i in range(2):
        if length & 255 << i * 8:
            scratch.append(length >> i * 8 & 255)
            scratch[0] |= 1 << 4 + i
    return bytes(scratch)