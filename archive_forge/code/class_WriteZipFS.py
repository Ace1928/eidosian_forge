from __future__ import print_function, unicode_literals
import sys
import typing
import six
import zipfile
from datetime import datetime
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_zip
from .enums import ResourceType, Seek
from .info import Info
from .iotools import RawWrapper
from .memoryfs import MemoryFS
from .opener import open_fs
from .path import dirname, forcedir, normpath, relpath
from .permissions import Permissions
from .time import datetime_to_epoch
from .wrapfs import WrapFS
@six.python_2_unicode_compatible
class WriteZipFS(WrapFS):
    """A writable zip file."""

    def __init__(self, file, compression=zipfile.ZIP_DEFLATED, encoding='utf-8', temp_fs='temp://__ziptemp__'):
        self._file = file
        self.compression = compression
        self.encoding = encoding
        self._temp_fs_url = temp_fs
        self._temp_fs = open_fs(temp_fs)
        self._meta = dict(self._temp_fs.getmeta())
        super(WriteZipFS, self).__init__(self._temp_fs)

    def __repr__(self):
        t = 'WriteZipFS({!r}, compression={!r}, encoding={!r}, temp_fs={!r})'
        return t.format(self._file, self.compression, self.encoding, self._temp_fs_url)

    def __str__(self):
        return "<zipfs-write '{}'>".format(self._file)

    def delegate_path(self, path):
        return (self._temp_fs, path)

    def delegate_fs(self):
        return self._temp_fs

    def close(self):
        if not self.isclosed():
            try:
                self.write_zip()
            finally:
                self._temp_fs.close()
        super(WriteZipFS, self).close()

    def write_zip(self, file=None, compression=None, encoding=None):
        """Write zip to a file.

        Arguments:
            file (str or io.IOBase, optional): Destination file, may be
                a file name or an open file handle.
            compression (int, optional): Compression to use (one of the
                constants defined in the `zipfile` module in the stdlib).
            encoding (str, optional): The character encoding to use
                (default uses the encoding defined in
                `~WriteZipFS.__init__`).

        Note:
            This is called automatically when the ZipFS is closed.

        """
        if not self.isclosed():
            write_zip(self._temp_fs, file or self._file, compression=compression or self.compression, encoding=encoding or self.encoding)