import sys
from threading import Lock
from typing import Union
from ._cffi_ppmd import ffi, lib
def _free(self):
    if self._finished:
        return
    self._finished = True
    lib.Ppmd8T_Free(self.ppmd, self.threadInfo, self._allocator)
    ffi.release(self.ppmd)
    self._release()