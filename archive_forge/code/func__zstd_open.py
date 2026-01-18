from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
def _zstd_open(filename: str, mode: Mode='r', *, level_or_option: int | dict | None=None, zstd_dict: pyzstd.ZstdDict | None=None) -> pyzstd.ZstdFile:
    return pyzstd.ZstdFile(filename, mode, level_or_option=level_or_option, zstd_dict=zstd_dict)