from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
def add_file_from_memory(self, entry_path, entry_size, entry_data, filetype=FileType.REGULAR_FILE, permission=436, **other_attributes):
    """"Add file from memory to archive.

        Args:
            entry_path (str | bytes): the file's path
            entry_size (int): the file's size, in bytes
            entry_data (bytes | Iterable[bytes]): the file's content
            filetype (int): see `libarchive.entry.ArchiveEntry.modify()`
            permission (int): see `libarchive.entry.ArchiveEntry.modify()`
            other_attributes: see `libarchive.entry.ArchiveEntry.modify()`
        """
    archive_pointer = self._pointer
    if isinstance(entry_data, bytes):
        entry_data = (entry_data,)
    elif isinstance(entry_data, str):
        raise TypeError('entry_data: expected bytes, got %r' % type(entry_data))
    entry = ArchiveEntry(pathname=entry_path, size=entry_size, filetype=filetype, perm=permission, header_codec=self.header_codec, **other_attributes)
    write_header(archive_pointer, entry._entry_p)
    for chunk in entry_data:
        if not chunk:
            break
        write_data(archive_pointer, chunk, len(chunk))
    write_finish_entry(archive_pointer)