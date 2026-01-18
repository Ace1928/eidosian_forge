import ctypes
import json
import os
import os.path
import shutil
import sys
from typing import Any, Dict, Optional
import warnings
def _diagnose_win32_dll_load() -> str:
    depends = _get_json_data('_depends.json')
    if depends is None:
        return ''
    from ctypes import wintypes
    kernel32 = ctypes.CDLL('kernel32')
    kernel32.GetModuleFileNameW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD]
    kernel32.GetModuleFileNameW.restype = wintypes.DWORD
    lines = ['', '', f'CUDA Path: {get_cuda_path()}', 'DLL dependencies:']
    filepath = ctypes.create_unicode_buffer(2 ** 15)
    for name in depends['depends']:
        try:
            dll = ctypes.CDLL(name)
            kernel32.GetModuleFileNameW(dll._handle, filepath, len(filepath))
            lines.append(f'  {name} -> {filepath.value}')
        except FileNotFoundError:
            lines.append(f'  {name} -> not found')
        except Exception as e:
            lines.append(f'  {name} -> error ({type(e).__name__}: {e})')
    return '\n'.join(lines)