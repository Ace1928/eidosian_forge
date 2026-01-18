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
class _ZipWriteFile(io.BufferedIOBase):

    def __init__(self, zf, zinfo, zip64):
        self._zinfo = zinfo
        self._zip64 = zip64
        self._zipfile = zf
        self._compressor = _get_compressor(zinfo.compress_type, zinfo._compresslevel)
        self._file_size = 0
        self._compress_size = 0
        self._crc = 0

    @property
    def _fileobj(self):
        return self._zipfile.fp

    def writable(self):
        return True

    def write(self, data):
        if self.closed:
            raise ValueError('I/O operation on closed file.')
        if isinstance(data, (bytes, bytearray)):
            nbytes = len(data)
        else:
            data = memoryview(data)
            nbytes = data.nbytes
        self._file_size += nbytes
        self._crc = crc32(data, self._crc)
        if self._compressor:
            data = self._compressor.compress(data)
            self._compress_size += len(data)
        self._fileobj.write(data)
        return nbytes

    def close(self):
        if self.closed:
            return
        try:
            super().close()
            if self._compressor:
                buf = self._compressor.flush()
                self._compress_size += len(buf)
                self._fileobj.write(buf)
                self._zinfo.compress_size = self._compress_size
            else:
                self._zinfo.compress_size = self._file_size
            self._zinfo.CRC = self._crc
            self._zinfo.file_size = self._file_size
            if not self._zip64:
                if self._file_size > ZIP64_LIMIT:
                    raise RuntimeError('File size too large, try using force_zip64')
                if self._compress_size > ZIP64_LIMIT:
                    raise RuntimeError('Compressed size too large, try using force_zip64')
            if self._zinfo.flag_bits & _MASK_USE_DATA_DESCRIPTOR:
                fmt = '<LLQQ' if self._zip64 else '<LLLL'
                self._fileobj.write(struct.pack(fmt, _DD_SIGNATURE, self._zinfo.CRC, self._zinfo.compress_size, self._zinfo.file_size))
                self._zipfile.start_dir = self._fileobj.tell()
            else:
                self._zipfile.start_dir = self._fileobj.tell()
                self._fileobj.seek(self._zinfo.header_offset)
                self._fileobj.write(self._zinfo.FileHeader(self._zip64))
                self._fileobj.seek(self._zipfile.start_dir)
            self._zipfile.filelist.append(self._zinfo)
            self._zipfile.NameToInfo[self._zinfo.filename] = self._zinfo
        finally:
            self._zipfile._writing = False