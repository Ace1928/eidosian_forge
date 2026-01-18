from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
class DeterministicGzipFile(gzip.GzipFile):
    """Deterministic variant of GzipFile

    This writer does not add filename information to the header, and defaults
    to a modification time (``mtime``) of 0 seconds.
    """

    def __init__(self, filename: str | None=None, mode: Mode | None=None, compresslevel: int=9, fileobj: io.FileIO | None=None, mtime: int=0):
        if mode is None:
            mode = 'rb'
        modestr: str = mode
        if 'b' not in modestr:
            modestr = f'{mode}b'
        if fileobj is None:
            if filename is None:
                raise TypeError('Must define either fileobj or filename')
            fileobj = self.myfileobj = ty.cast(io.FileIO, open(filename, modestr))
        return super().__init__(filename='', mode=modestr, compresslevel=compresslevel, fileobj=fileobj, mtime=mtime)