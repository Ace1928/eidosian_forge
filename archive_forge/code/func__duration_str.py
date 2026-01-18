import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
@property
def _duration_str(self):
    hours, rest = divmod(self.duration, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours >= 1:
        duration = '{0:.0g}:{1:02.0g}:{2:05.3f} h'.format(hours, minutes, seconds)
    elif minutes >= 1:
        duration = '{0:02.0g}:{1:05.3f} min'.format(minutes, seconds)
    elif seconds <= 1:
        duration = '{0:d} samples'.format(self.frames)
    else:
        duration = '{0:.3f} s'.format(seconds)
    return duration