import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _create_empty_array(self, frames, always_2d, dtype):
    """Create an empty array with appropriate shape."""
    import numpy as np
    if always_2d or self.channels > 1:
        shape = (frames, self.channels)
    else:
        shape = (frames,)
    return np.empty(shape, dtype, order='C')