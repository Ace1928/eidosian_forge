import os
from .dependencies import ctypes
class _MsvcrtDLL(object):
    """Helper class to manage the interface with the MSVCRT runtime"""

    def __init__(self, name):
        self._libname = name
        if name is None:
            self._loaded = False
        else:
            self._loaded = None
        self.dll = None

    def available(self):
        if self._loaded is not None:
            return self._loaded
        self._loaded, self.dll = _load_dll(self._libname)
        if not self._loaded:
            return self._loaded
        self.putenv_s = self.dll._putenv_s
        self.putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.putenv_s.restype = ctypes.c_int
        self.wputenv_s = self.dll._wputenv_s
        self.wputenv_s.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        self.wputenv_s.restype = ctypes.c_int
        self.getenv = self.dll.getenv
        self.getenv.argtypes = [ctypes.c_char_p]
        self.getenv.restype = ctypes.c_char_p
        self.wgetenv = self.dll._wgetenv
        self.wgetenv.argtypes = [ctypes.c_wchar_p]
        self.wgetenv.restype = ctypes.c_wchar_p
        return self._loaded

    def get_env_dict(self):
        if not self.available():
            return None
        try:
            envp = ctypes.POINTER(ctypes.c_wchar_p).in_dll(self.dll, '_wenviron')
            if not envp.contents:
                envp = None
        except ValueError:
            envp = None
        if envp is None:
            try:
                envp = ctypes.POINTER(ctypes.c_char_p).in_dll(self.dll, '_environ')
                if not envp.contents:
                    return None
            except ValueError:
                return None
        ans = {}
        size = 0
        for line in envp:
            if not line:
                break
            size += len(line)
            if len(line) == 0:
                raise ValueError('Error processing MSVCRT _environ: 0-length string encountered')
            if size > 32767:
                raise ValueError('Error processing MSVCRT _environ: exceeded max environment block size (32767)')
            key, val = line.split('=', 1)
            ans[key] = val
        return ans