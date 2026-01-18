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
def _obj(self) -> OldUnpackedObject:
    """Return the decompressed chunks, or (delta base, delta chunks)."""
    if self.pack_type_num in DELTA_TYPES:
        assert isinstance(self.delta_base, (bytes, int))
        return (self.delta_base, self.decomp_chunks)
    else:
        return self.decomp_chunks