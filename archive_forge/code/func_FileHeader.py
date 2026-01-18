import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def FileHeader(self, zip64=None):
    """Return the per-file header as a bytes object.

        When the optional zip64 arg is None rather than a bool, we will
        decide based upon the file_size and compress_size, if known,
        False otherwise.
        """
    dt = self.date_time
    dosdate = dt[0] - 1980 << 9 | dt[1] << 5 | dt[2]
    dostime = dt[3] << 11 | dt[4] << 5 | dt[5] // 2
    if self.flag_bits & _MASK_USE_DATA_DESCRIPTOR:
        CRC = compress_size = file_size = 0
    else:
        CRC = self.CRC
        compress_size = self.compress_size
        file_size = self.file_size
    extra = self.extra
    min_version = 0
    if zip64 is None:
        zip64 = file_size > ZIP64_LIMIT or compress_size > ZIP64_LIMIT
    if zip64:
        fmt = '<HHQQ'
        extra = extra + struct.pack(fmt, 1, struct.calcsize(fmt) - 4, file_size, compress_size)
        file_size = 4294967295
        compress_size = 4294967295
        min_version = ZIP64_VERSION
    if self.compress_type == ZIP_BZIP2:
        min_version = max(BZIP2_VERSION, min_version)
    elif self.compress_type == ZIP_LZMA:
        min_version = max(LZMA_VERSION, min_version)
    self.extract_version = max(min_version, self.extract_version)
    self.create_version = max(min_version, self.create_version)
    filename, flag_bits = self._encodeFilenameFlags()
    header = struct.pack(structFileHeader, stringFileHeader, self.extract_version, self.reserved, flag_bits, self.compress_type, dostime, dosdate, CRC, compress_size, file_size, len(filename), len(extra))
    return header + filename + extra