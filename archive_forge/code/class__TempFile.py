from __future__ import annotations
import logging # isort:skip
import io
import os
from contextlib import contextmanager
from os.path import abspath, expanduser, splitext
from tempfile import mkstemp
from typing import (
from ..core.types import PathLike
from ..document import Document
from ..embed import file_html
from ..resources import INLINE, Resources
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
class _TempFile:
    _closed: bool = False
    fd: int
    path: str

    def __init__(self, *, prefix: str='tmp', suffix: str='') -> None:
        try:
            self.fd, self.path = mkstemp(prefix=prefix, suffix=suffix, dir=os.getcwd())
        except OSError:
            self.fd, self.path = mkstemp(prefix=prefix, suffix=suffix)

    def __enter__(self) -> _TempFile:
        return self

    def __exit__(self, exc: Any, value: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass
        self._closed = True