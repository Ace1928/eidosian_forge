import contextlib
from functools import wraps
import os
import os.path as osp
import struct
import tempfile
from types import TracebackType
from typing import Any, Callable, TYPE_CHECKING, Optional, Type
from git.types import Literal, PathLike, _T
class TemporaryFileSwap:
    """Utility class moving a file to a temporary location within the same directory
    and moving it back on to where on object deletion."""
    __slots__ = ('file_path', 'tmp_file_path')

    def __init__(self, file_path: PathLike) -> None:
        self.file_path = file_path
        dirname, basename = osp.split(file_path)
        fd, self.tmp_file_path = tempfile.mkstemp(prefix=basename, dir=dirname)
        os.close(fd)
        with contextlib.suppress(OSError):
            os.replace(self.file_path, self.tmp_file_path)

    def __enter__(self) -> 'TemporaryFileSwap':
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> Literal[False]:
        if osp.isfile(self.tmp_file_path):
            os.replace(self.tmp_file_path, self.file_path)
        return False