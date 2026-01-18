from typing import Optional
import types, gc, os, time, re, platform
import torch
from torch.nn import functional as F
def LoadPreCompileLibrary(file):
    import importlib
    import os
    import torch
    lib_dir = os.path.dirname(__file__)
    if os.name == 'nt':
        import ctypes
        import sys
        kernel32 = ctypes.WinDLL('kernel32.dll', use_last_error=True)
        with_load_library_flags = hasattr(kernel32, 'AddDllDirectory')
        prev_error_mode = kernel32.SetErrorMode(1)
        if with_load_library_flags:
            kernel32.AddDllDirectory.restype = ctypes.c_void_p
        if sys.version_info >= (3, 8):
            os.add_dll_directory(lib_dir)
        elif with_load_library_flags:
            res = kernel32.AddDllDirectory(lib_dir)
            if res is None:
                err = ctypes.WinError(ctypes.get_last_error())
                err.strerror += f' Error adding "{lib_dir}" to the DLL directories.'
                raise ValueError(err)
        kernel32.SetErrorMode(prev_error_mode)
    loader_details = (importlib.machinery.ExtensionFileLoader, importlib.machinery.EXTENSION_SUFFIXES)
    extfinder = importlib.machinery.FileFinder(lib_dir, loader_details)
    ext_specs = extfinder.find_spec(file)
    if ext_specs is None:
        return False
    try:
        torch.ops.load_library(ext_specs.origin)
    except OSError as exc:
        return False
    return True