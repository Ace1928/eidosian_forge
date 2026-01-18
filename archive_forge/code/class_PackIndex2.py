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
class PackIndex2(FilePackIndex):
    """Version 2 Pack Index file."""

    def __init__(self, filename: str, file=None, contents=None, size=None) -> None:
        super().__init__(filename, file, contents, size)
        if self._contents[:4] != b'\xfftOc':
            raise AssertionError('Not a v2 pack index file')
        self.version, = unpack_from(b'>L', self._contents, 4)
        if self.version != 2:
            raise AssertionError('Version was %d' % self.version)
        self._fan_out_table = self._read_fan_out_table(8)
        self._name_table_offset = 8 + 256 * 4
        self._crc32_table_offset = self._name_table_offset + 20 * len(self)
        self._pack_offset_table_offset = self._crc32_table_offset + 4 * len(self)
        self._pack_offset_largetable_offset = self._pack_offset_table_offset + 4 * len(self)

    def _unpack_entry(self, i):
        return (self._unpack_name(i), self._unpack_offset(i), self._unpack_crc32_checksum(i))

    def _unpack_name(self, i):
        offset = self._name_table_offset + i * 20
        return self._contents[offset:offset + 20]

    def _unpack_offset(self, i):
        offset = self._pack_offset_table_offset + i * 4
        offset = unpack_from('>L', self._contents, offset)[0]
        if offset & 2 ** 31:
            offset = self._pack_offset_largetable_offset + (offset & 2 ** 31 - 1) * 8
            offset = unpack_from('>Q', self._contents, offset)[0]
        return offset

    def _unpack_crc32_checksum(self, i):
        return unpack_from('>L', self._contents, self._crc32_table_offset + i * 4)[0]