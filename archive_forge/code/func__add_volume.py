import contextlib
import io
import os
import pathlib
from mmap import mmap
from typing import Any, Container, List, Optional, Union
from .stat import stat_result
def _add_volume(self):
    num = len(self._fileinfo) + self._start - 1
    if self._hex:
        last = self._fileinfo[-1].filename
        last_ext = '.{num:0{ext_digit}x}'.format(num=num, ext_digit=self._digits)
        assert last.suffix.endswith(last_ext)
        next_ext = '.{num:0{ext_digit}x}'.format(num=num + 1, ext_digit=self._digits)
        next = last.with_suffix(next_ext)
    else:
        last = self._fileinfo[-1].filename
        last_ext = '.{num:0{ext_digit}d}'.format(num=num, ext_digit=self._digits)
        assert last.suffix.endswith(last_ext)
        next_ext = '.{num:0{ext_digit}d}'.format(num=num + 1, ext_digit=self._digits)
        next = last.with_suffix(next_ext)
    self._files.append(io.open(next, self._mode))
    stat = os.stat(next)
    self._fileinfo.append(_FileInfo(next, stat, self._volume_size))
    pos = self._positions[-1]
    if pos != self._position:
        self._positions[-1] = self._position
    self._positions.append(self._positions[-1] + self._volume_size)