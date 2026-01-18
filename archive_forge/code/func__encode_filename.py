import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
def _encode_filename(filename):
    """Return a byte string suitable for a filename.

    Unicode is encoded using an encoding adapted to what both cairo and the
    filesystem want.

    """
    errors = 'ignore' if os.name == 'nt' else 'replace'
    if not isinstance(filename, bytes):
        if os.name == 'nt' and cairo.cairo_version() >= 11510:
            filename = filename.encode('utf-8', errors=errors)
        else:
            try:
                filename = filename.encode(sys.getfilesystemencoding())
            except UnicodeEncodeError:
                filename = filename.encode('ascii', errors=errors)
    return ffi.new('char[]', filename)