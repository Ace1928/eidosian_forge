import atexit
import logging
import os
import time
from concurrent.futures import Future
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, List, Optional, Union
from .hf_api import IGNORE_GIT_FOLDER_PATTERNS, CommitInfo, CommitOperationAdd, HfApi
from .utils import filter_repo_objects
class PartialFileIO(BytesIO):
    """A file-like object that reads only the first part of a file.

    Useful to upload a file to the Hub when the user might still be appending data to it. Only the first part of the
    file is uploaded (i.e. the part that was available when the filesystem was first scanned).

    In practice, only used internally by the CommitScheduler to regularly push a folder to the Hub with minimal
    disturbance for the user. The object is passed to `CommitOperationAdd`.

    Only supports `read`, `tell` and `seek` methods.

    Args:
        file_path (`str` or `Path`):
            Path to the file to read.
        size_limit (`int`):
            The maximum number of bytes to read from the file. If the file is larger than this, only the first part
            will be read (and uploaded).
    """

    def __init__(self, file_path: Union[str, Path], size_limit: int) -> None:
        self._file_path = Path(file_path)
        self._file = self._file_path.open('rb')
        self._size_limit = min(size_limit, os.fstat(self._file.fileno()).st_size)

    def __del__(self) -> None:
        self._file.close()
        return super().__del__()

    def __repr__(self) -> str:
        return f'<PartialFileIO file_path={self._file_path} size_limit={self._size_limit}>'

    def __len__(self) -> int:
        return self._size_limit

    def __getattribute__(self, name: str):
        if name.startswith('_') or name in ('read', 'tell', 'seek'):
            return super().__getattribute__(name)
        raise NotImplementedError(f"PartialFileIO does not support '{name}'.")

    def tell(self) -> int:
        """Return the current file position."""
        return self._file.tell()

    def seek(self, __offset: int, __whence: int=SEEK_SET) -> int:
        """Change the stream position to the given offset.

        Behavior is the same as a regular file, except that the position is capped to the size limit.
        """
        if __whence == SEEK_END:
            __offset = len(self) + __offset
            __whence = SEEK_SET
        pos = self._file.seek(__offset, __whence)
        if pos > self._size_limit:
            return self._file.seek(self._size_limit)
        return pos

    def read(self, __size: Optional[int]=-1) -> bytes:
        """Read at most `__size` bytes from the file.

        Behavior is the same as a regular file, except that it is capped to the size limit.
        """
        current = self._file.tell()
        if __size is None or __size < 0:
            truncated_size = self._size_limit - current
        else:
            truncated_size = min(__size, self._size_limit - current)
        return self._file.read(truncated_size)