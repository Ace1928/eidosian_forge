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
def _write_end_record(self):
    for zinfo in self.filelist:
        dt = zinfo.date_time
        dosdate = dt[0] - 1980 << 9 | dt[1] << 5 | dt[2]
        dostime = dt[3] << 11 | dt[4] << 5 | dt[5] // 2
        extra = []
        if zinfo.file_size > ZIP64_LIMIT or zinfo.compress_size > ZIP64_LIMIT:
            extra.append(zinfo.file_size)
            extra.append(zinfo.compress_size)
            file_size = 4294967295
            compress_size = 4294967295
        else:
            file_size = zinfo.file_size
            compress_size = zinfo.compress_size
        if zinfo.header_offset > ZIP64_LIMIT:
            extra.append(zinfo.header_offset)
            header_offset = 4294967295
        else:
            header_offset = zinfo.header_offset
        extra_data = zinfo.extra
        min_version = 0
        if extra:
            extra_data = _strip_extra(extra_data, (1,))
            extra_data = struct.pack('<HH' + 'Q' * len(extra), 1, 8 * len(extra), *extra) + extra_data
            min_version = ZIP64_VERSION
        if zinfo.compress_type == ZIP_BZIP2:
            min_version = max(BZIP2_VERSION, min_version)
        elif zinfo.compress_type == ZIP_LZMA:
            min_version = max(LZMA_VERSION, min_version)
        extract_version = max(min_version, zinfo.extract_version)
        create_version = max(min_version, zinfo.create_version)
        filename, flag_bits = zinfo._encodeFilenameFlags()
        centdir = struct.pack(structCentralDir, stringCentralDir, create_version, zinfo.create_system, extract_version, zinfo.reserved, flag_bits, zinfo.compress_type, dostime, dosdate, zinfo.CRC, compress_size, file_size, len(filename), len(extra_data), len(zinfo.comment), 0, zinfo.internal_attr, zinfo.external_attr, header_offset)
        self.fp.write(centdir)
        self.fp.write(filename)
        self.fp.write(extra_data)
        self.fp.write(zinfo.comment)
    pos2 = self.fp.tell()
    centDirCount = len(self.filelist)
    centDirSize = pos2 - self.start_dir
    centDirOffset = self.start_dir
    requires_zip64 = None
    if centDirCount > ZIP_FILECOUNT_LIMIT:
        requires_zip64 = 'Files count'
    elif centDirOffset > ZIP64_LIMIT:
        requires_zip64 = 'Central directory offset'
    elif centDirSize > ZIP64_LIMIT:
        requires_zip64 = 'Central directory size'
    if requires_zip64:
        if not self._allowZip64:
            raise LargeZipFile(requires_zip64 + ' would require ZIP64 extensions')
        zip64endrec = struct.pack(structEndArchive64, stringEndArchive64, 44, 45, 45, 0, 0, centDirCount, centDirCount, centDirSize, centDirOffset)
        self.fp.write(zip64endrec)
        zip64locrec = struct.pack(structEndArchive64Locator, stringEndArchive64Locator, 0, pos2, 1)
        self.fp.write(zip64locrec)
        centDirCount = min(centDirCount, 65535)
        centDirSize = min(centDirSize, 4294967295)
        centDirOffset = min(centDirOffset, 4294967295)
    endrec = struct.pack(structEndArchive, stringEndArchive, 0, 0, centDirCount, centDirCount, centDirSize, centDirOffset, len(self._comment))
    self.fp.write(endrec)
    self.fp.write(self._comment)
    if self.mode == 'a':
        self.fp.truncate()
    self.fp.flush()