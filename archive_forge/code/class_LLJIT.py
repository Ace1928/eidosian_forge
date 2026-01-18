import ctypes
from ctypes import POINTER, c_bool, c_char_p, c_uint8, c_uint64, c_size_t
from llvmlite.binding import ffi, targets
class LLJIT(ffi.ObjectRef):
    """
    A OrcJIT-based LLVM JIT engine that can compile and run LLVM IR as a
    collection of JITted dynamic libraries

    The C++ OrcJIT API has a lot of memory ownership patterns that do not work
    with Python. This API attempts to provide ones that are safe at the expense
    of some features. Each LLJIT instance is a collection of JIT-compiled
    libraries. In the C++ API, there is a "main" library; this API does not
    provide access to the main library. Use the JITLibraryBuilder to create a
    new named library instead.
    """

    def __init__(self, ptr):
        self._td = None
        ffi.ObjectRef.__init__(self, ptr)

    def lookup(self, dylib, fn):
        """
        Find a function in this dynamic library and construct a new tracking
        object for it

        If the library or function do not exist, an exception will occur.

        Parameters
        ----------
        dylib : str or None
           the name of the library containing the symbol
        fn : str
           the name of the function to get
        """
        assert not self.closed, 'Cannot lookup in closed JIT'
        address = ctypes.c_uint64()
        with ffi.OutputString() as outerr:
            tracker = ffi.lib.LLVMPY_LLJITLookup(self, dylib.encode('utf-8'), fn.encode('utf-8'), ctypes.byref(address), outerr)
            if not tracker:
                raise RuntimeError(str(outerr))
        return ResourceTracker(tracker, dylib, {fn: address.value})

    @property
    def target_data(self):
        """
        The TargetData for this LLJIT instance.
        """
        if self._td is not None:
            return self._td
        ptr = ffi.lib.LLVMPY_LLJITGetDataLayout(self)
        self._td = targets.TargetData(ptr)
        self._td._owned = True
        return self._td

    def _dispose(self):
        if self._td is not None:
            self._td.detach()
        self._capi.LLVMPY_LLJITDispose(self)