import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _array_io(self, action, array, frames):
    """Check array and call low-level IO function."""
    if array.ndim not in (1, 2) or (array.ndim == 1 and self.channels != 1) or (array.ndim == 2 and array.shape[1] != self.channels):
        raise ValueError('Invalid shape: {0!r}'.format(array.shape))
    if not array.flags.c_contiguous:
        raise ValueError('Data must be C-contiguous')
    ctype = self._check_dtype(array.dtype.name)
    assert array.dtype.itemsize == _ffi.sizeof(ctype)
    cdata = _ffi.cast(ctype + '*', array.__array_interface__['data'][0])
    return self._cdata_io(action, cdata, ctype, frames)