import platform
from ctypes import (POINTER, c_char_p, c_bool, c_void_p,
from llvmlite.binding import ffi, targets, object_file
def _raw_object_cache_getbuffer(self, data):
    """
        Low-level getbuffer hook.
        """
    if self._object_cache_getbuffer is None:
        return
    module_ptr = data.contents.module_ptr
    module = self._find_module_ptr(module_ptr)
    if module is None:
        raise RuntimeError('object compilation notification for unknown module %s' % (module_ptr,))
    buf = self._object_cache_getbuffer(module)
    if buf is not None:
        data[0].buf_ptr = ffi.lib.LLVMPY_CreateByteString(buf, len(buf))
        data[0].buf_len = len(buf)