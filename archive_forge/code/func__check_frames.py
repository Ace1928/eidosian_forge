import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _check_frames(self, frames, fill_value):
    """Reduce frames to no more than are available in the file."""
    if self.seekable():
        remaining_frames = self.frames - self.tell()
        if frames < 0 or (frames > remaining_frames and fill_value is None):
            frames = remaining_frames
    elif frames < 0:
        raise ValueError('frames must be specified for non-seekable files')
    return frames