import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _update_frames(self, written):
    """Update self.frames after writing."""
    if self.seekable():
        curr = self.tell()
        self._info.frames = self.seek(0, SEEK_END)
        self.seek(curr, SEEK_SET)
    else:
        self._info.frames += written