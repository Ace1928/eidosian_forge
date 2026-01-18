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
def _walk_ref_chains(self):
    if not self._resolve_ext_ref:
        self._ensure_no_pending()
        return
    for base_sha, pending in sorted(self._pending_ref.items()):
        if base_sha not in self._pending_ref:
            continue
        try:
            type_num, chunks = self._resolve_ext_ref(base_sha)
        except KeyError:
            continue
        self._ext_refs.append(base_sha)
        self._pending_ref.pop(base_sha)
        for new_offset in pending:
            yield from self._follow_chain(new_offset, type_num, chunks)
    self._ensure_no_pending()