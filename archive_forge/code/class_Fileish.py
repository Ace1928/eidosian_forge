from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
@ty.runtime_checkable
class Fileish(ty.Protocol):

    def read(self, size: int=-1, /) -> bytes:
        ...

    def write(self, b: bytes, /) -> int | None:
        ...