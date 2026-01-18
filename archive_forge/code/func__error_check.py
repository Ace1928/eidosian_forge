import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _error_check(err, prefix=''):
    """Raise LibsndfileError if there is an error."""
    if err != 0:
        raise LibsndfileError(err, prefix=prefix)