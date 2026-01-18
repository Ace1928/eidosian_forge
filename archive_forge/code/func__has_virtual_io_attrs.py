import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _has_virtual_io_attrs(file, mode_int):
    """Check if file has all the necessary attributes for virtual IO."""
    readonly = mode_int == _snd.SFM_READ
    writeonly = mode_int == _snd.SFM_WRITE
    return all([hasattr(file, 'seek'), hasattr(file, 'tell'), hasattr(file, 'write') or readonly, hasattr(file, 'read') or hasattr(file, 'readinto') or writeonly])