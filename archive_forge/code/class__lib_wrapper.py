import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
class _lib_wrapper(object):
    """Wrap libllvmlite with a lock such that only one thread may access it at
    a time.

    This class duck-types a CDLL.
    """
    __slots__ = ['_lib_handle', '_fntab', '_lock']

    def __init__(self):
        self._lib_handle = None
        self._fntab = {}
        self._lock = _LLVMLock()

    def _load_lib(self):
        try:
            with _suppress_cleanup_errors(_importlib_resources_path(__name__.rpartition('.')[0], get_library_name())) as lib_path:
                self._lib_handle = ctypes.CDLL(str(lib_path))
                _ = self._lib_handle.LLVMPY_GetVersionInfo()
        except (OSError, AttributeError) as e:
            raise OSError('Could not find/load shared object file') from e

    @property
    def _lib(self):
        if not self._lib_handle:
            self._load_lib()
        return self._lib_handle

    def __getattr__(self, name):
        try:
            return self._fntab[name]
        except KeyError:
            cfn = getattr(self._lib, name)
            wrapped = _lib_fn_wrapper(self._lock, cfn)
            self._fntab[name] = wrapped
            return wrapped

    @property
    def _name(self):
        """The name of the library passed in the CDLL constructor.

        For duck-typing a ctypes.CDLL
        """
        return self._lib._name

    @property
    def _handle(self):
        """The system handle used to access the library.

        For duck-typing a ctypes.CDLL
        """
        return self._lib._handle