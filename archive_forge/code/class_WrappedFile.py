import os
from typing import Optional
import fsspec
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.utils import DEFAULT_BLOCK_SIZE
class WrappedFile:

    def __init__(self, file_):
        self._file = file_

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self._file.__exit__(*args, **kwargs)

    def __iter__(self):
        return iter(self._file)

    def __next__(self):
        return next(self._file)

    def __getattr__(self, attr):
        return getattr(self._file, attr)