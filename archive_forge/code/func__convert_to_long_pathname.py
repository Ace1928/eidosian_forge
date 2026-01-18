import os
import sys
import tempfile
from IPython.core.compilerop import CachingCompiler
def _convert_to_long_pathname(filename):
    buf = ctypes.create_unicode_buffer(MAX_PATH)
    rv = _GetLongPathName(filename, buf, MAX_PATH)
    if rv != 0 and rv <= MAX_PATH:
        filename = buf.value
    return filename