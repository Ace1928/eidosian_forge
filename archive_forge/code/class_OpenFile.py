from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
class OpenFile:
    """
    File-like object to be used in a context

    Can layer (buffered) text-mode and compression over any file-system, which
    are typically binary-only.

    These instances are safe to serialize, as the low-level file object
    is not created until invoked using ``with``.

    Parameters
    ----------
    fs: FileSystem
        The file system to use for opening the file. Should be a subclass or duck-type
        with ``fsspec.spec.AbstractFileSystem``
    path: str
        Location to open
    mode: str like 'rb', optional
        Mode of the opened file
    compression: str or None, optional
        Compression to apply
    encoding: str or None, optional
        The encoding to use if opened in text mode.
    errors: str or None, optional
        How to handle encoding errors if opened in text mode.
    newline: None or str
        Passed to TextIOWrapper in text mode, how to handle line endings.
    autoopen: bool
        If True, calls open() immediately. Mostly used by pickle
    pos: int
        If given and autoopen is True, seek to this location immediately
    """

    def __init__(self, fs, path, mode='rb', compression=None, encoding=None, errors=None, newline=None):
        self.fs = fs
        self.path = path
        self.mode = mode
        self.compression = get_compression(path, compression)
        self.encoding = encoding
        self.errors = errors
        self.newline = newline
        self.fobjects = []

    def __reduce__(self):
        return (OpenFile, (self.fs, self.path, self.mode, self.compression, self.encoding, self.errors, self.newline))

    def __repr__(self):
        return f"<OpenFile '{self.path}'>"

    def __enter__(self):
        mode = self.mode.replace('t', '').replace('b', '') + 'b'
        f = self.fs.open(self.path, mode=mode)
        self.fobjects = [f]
        if self.compression is not None:
            compress = compr[self.compression]
            f = compress(f, mode=mode[0])
            self.fobjects.append(f)
        if 'b' not in self.mode:
            f = PickleableTextIOWrapper(f, encoding=self.encoding, errors=self.errors, newline=self.newline)
            self.fobjects.append(f)
        return self.fobjects[-1]

    def __exit__(self, *args):
        self.close()

    @property
    def full_name(self):
        return _unstrip_protocol(self.path, self.fs)

    def open(self):
        """Materialise this as a real open file without context

        The OpenFile object should be explicitly closed to avoid enclosed file
        instances persisting. You must, therefore, keep a reference to the OpenFile
        during the life of the file-like it generates.
        """
        return self.__enter__()

    def close(self):
        """Close all encapsulated file objects"""
        for f in reversed(self.fobjects):
            if 'r' not in self.mode and (not f.closed):
                f.flush()
            f.close()
        self.fobjects.clear()