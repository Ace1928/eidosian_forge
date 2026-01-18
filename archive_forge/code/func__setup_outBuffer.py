import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def _setup_outBuffer(self):
    out_buf = _new_nonzero('OutBuffer *')
    if out_buf == ffi.NULL:
        raise MemoryError
    out = _BlocksOutputBuffer()
    out.initAndGrow(out_buf, -1)
    return (out, out_buf)