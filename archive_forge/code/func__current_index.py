import contextlib
import io
import os
import pathlib
from mmap import mmap
from typing import Any, Container, List, Optional, Union
from .stat import stat_result
def _current_index(self):
    for i in range(len(self._positions) - 1):
        if self._positions[i] <= self._position < self._positions[i + 1]:
            pos = self._files[i].tell()
            offset = self._position - self._positions[i]
            if pos != offset:
                self._files[i].seek(offset, io.SEEK_SET)
            return i
    return len(self._files) - 1