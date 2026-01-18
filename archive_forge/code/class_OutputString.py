import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
class OutputString(object):
    """
    Object for managing the char* output of LLVM APIs.
    """
    _as_parameter_ = _DeadPointer()

    @classmethod
    def from_return(cls, ptr):
        """Constructing from a pointer returned from the C-API.
        The pointer must be allocated with LLVMPY_CreateString.

        Note
        ----
        Because ctypes auto-converts *restype* of *c_char_p* into a python
        string, we must use *c_void_p* to obtain the raw pointer.
        """
        return cls(init=ctypes.cast(ptr, ctypes.c_char_p))

    def __init__(self, owned=True, init=None):
        self._ptr = init if init is not None else ctypes.c_char_p(None)
        self._as_parameter_ = ctypes.byref(self._ptr)
        self._owned = owned

    def close(self):
        if self._ptr is not None:
            if self._owned:
                lib.LLVMPY_DisposeString(self._ptr)
            self._ptr = None
            del self._as_parameter_

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self, _is_shutting_down=_is_shutting_down):
        if not _is_shutting_down():
            if self.close is not None:
                self.close()

    def __str__(self):
        if self._ptr is None:
            return '<dead OutputString>'
        s = self._ptr.value
        assert s is not None
        return _decode_string(s)

    def __bool__(self):
        return bool(self._ptr)
    __nonzero__ = __bool__

    @property
    def bytes(self):
        """Get the raw bytes of content of the char pointer.
        """
        return self._ptr.value