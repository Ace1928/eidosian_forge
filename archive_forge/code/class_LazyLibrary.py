import threading
import importlib
from cffi import FFI
class LazyLibrary(object):

    def __init__(self, ffi, libname):
        self._ffi = ffi
        self._libname = libname
        self._lib = None
        self._lock = threading.Lock()

    def __getattr__(self, name):
        if self._lib is None:
            with self._lock:
                if self._lib is None:
                    self._lib = self._ffi.dlopen(self._libname)
        return getattr(self._lib, name)