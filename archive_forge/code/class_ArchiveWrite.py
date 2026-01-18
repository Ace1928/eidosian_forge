from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
class ArchiveWrite:

    def __init__(self, archive_p, header_codec='utf-8'):
        self._pointer = archive_p
        self.header_codec = header_codec

    def add_entries(self, entries):
        """Add the given entries to the archive.
        """
        write_p = self._pointer
        for entry in entries:
            write_header(write_p, entry._entry_p)
            for block in entry.get_blocks():
                write_data(write_p, block, len(block))
            write_finish_entry(write_p)

    def add_files(self, *paths, flags=0, lookup=False, pathname=None, recursive=True, **attributes):
        """Read files through the OS and add them to the archive.

        Args:
            paths (str): the paths of the files to add to the archive
            flags (int):
                passed to the C function `archive_read_disk_set_behavior`;
                use the `libarchive.flags.READDISK_*` constants
            lookup (bool):
                when True, the C function `archive_read_disk_set_standard_lookup`
                is called to enable the lookup of user and group names
            pathname (str | None):
                the path of the file in the archive, defaults to the source path
            recursive (bool):
                when False, if a path in `paths` is a directory,
                only the directory itself is added.
            attributes (dict): passed to `ArchiveEntry.modify()`

        Raises:
            ArchiveError: if a file doesn't exist or can't be accessed, or if
                          adding it to the archive fails
        """
        write_p = self._pointer
        block_size = ffi.write_get_bytes_per_block(write_p)
        if block_size <= 0:
            block_size = 10240
        entry = ArchiveEntry(header_codec=self.header_codec)
        entry_p = entry._entry_p
        destination_path = attributes.pop('pathname', None)
        for path in paths:
            with new_archive_read_disk(path, flags, lookup) as read_p:
                while 1:
                    r = read_next_header2(read_p, entry_p)
                    if r == ARCHIVE_EOF:
                        break
                    entry_path = entry.pathname
                    if destination_path:
                        if entry_path == path:
                            entry_path = destination_path
                        else:
                            assert entry_path.startswith(path)
                            entry_path = join(destination_path, entry_path[len(path):].lstrip('/'))
                    entry.pathname = entry_path.lstrip('/')
                    if attributes:
                        entry.modify(**attributes)
                    read_disk_descend(read_p)
                    write_header(write_p, entry_p)
                    if entry.isreg:
                        with open(entry_sourcepath(entry_p), 'rb') as f:
                            while 1:
                                data = f.read(block_size)
                                if not data:
                                    break
                                write_data(write_p, data, len(data))
                    write_finish_entry(write_p)
                    entry_clear(entry_p)
                    if not recursive:
                        break

    def add_file(self, path, **kw):
        """Single-path alias of `add_files()`"""
        return self.add_files(path, **kw)

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