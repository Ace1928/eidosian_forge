from contextlib import contextmanager
from ctypes import cast, c_void_p, POINTER, create_string_buffer
from os import fstat, stat
from . import ffi
from .ffi import (
from .entry import ArchiveEntry, PassedArchiveEntry
class ArchiveRead:

    def __init__(self, archive_p, header_codec='utf-8'):
        self._pointer = archive_p
        self.header_codec = header_codec

    def __iter__(self):
        """Iterates through an archive's entries.
        """
        archive_p = self._pointer
        header_codec = self.header_codec
        read_next_header2 = ffi.read_next_header2
        while 1:
            entry = ArchiveEntry(archive_p, header_codec)
            r = read_next_header2(archive_p, entry._entry_p)
            if r == ARCHIVE_EOF:
                return
            yield entry
            entry.__class__ = PassedArchiveEntry

    @property
    def bytes_read(self):
        return ffi.filter_bytes(self._pointer, -1)

    @property
    def filter_names(self):
        count = ffi.filter_count(self._pointer)
        return [ffi.filter_name(self._pointer, i) for i in range(count - 1)]

    @property
    def format_name(self):
        return ffi.format_name(self._pointer)