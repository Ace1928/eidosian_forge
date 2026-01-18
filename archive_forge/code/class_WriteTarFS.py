from __future__ import print_function, unicode_literals
import typing
from typing import IO, cast
import os
import six
import tarfile
from collections import OrderedDict
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_tar
from .enums import ResourceType
from .errors import IllegalBackReference, NoURL
from .info import Info
from .iotools import RawWrapper
from .opener import open_fs
from .path import basename, frombase, isbase, normpath, parts, relpath
from .permissions import Permissions
from .wrapfs import WrapFS
@six.python_2_unicode_compatible
class WriteTarFS(WrapFS):
    """A writable tar file."""

    def __init__(self, file, compression=None, encoding='utf-8', temp_fs='temp://__tartemp__'):
        self._file = file
        self.compression = compression
        self.encoding = encoding
        self._temp_fs_url = temp_fs
        self._temp_fs = open_fs(temp_fs)
        self._meta = dict(self._temp_fs.getmeta())
        super(WriteTarFS, self).__init__(self._temp_fs)

    def __repr__(self):
        t = 'WriteTarFS({!r}, compression={!r}, encoding={!r}, temp_fs={!r})'
        return t.format(self._file, self.compression, self.encoding, self._temp_fs_url)

    def __str__(self):
        return "<TarFS-write '{}'>".format(self._file)

    def delegate_path(self, path):
        return (self._temp_fs, path)

    def delegate_fs(self):
        return self._temp_fs

    def close(self):
        if not self.isclosed():
            try:
                self.write_tar()
            finally:
                self._temp_fs.close()
        super(WriteTarFS, self).close()

    def write_tar(self, file=None, compression=None, encoding=None):
        """Write tar to a file.

        Arguments:
            file (str or io.IOBase, optional): Destination file, may be
                a file name or an open file object.
            compression (str, optional): Compression to use (one of
                the constants defined in `tarfile` in the stdlib).
            encoding (str, optional): The character encoding to use
                (default uses the encoding defined in
                `~WriteTarFS.__init__`).

        Note:
            This is called automatically when the TarFS is closed.

        """
        if not self.isclosed():
            write_tar(self._temp_fs, file or self._file, compression=compression or self.compression, encoding=encoding or self.encoding)