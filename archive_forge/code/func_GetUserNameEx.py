from __future__ import annotations
import errno
import os
import site
import stat
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional
import platformdirs
from .utils import deprecation
def GetUserNameEx(NameFormat: Any) -> Any:
    nSize = ctypes.pointer(ctypes.c_ulong(0))
    try:
        secur32.GetUserNameExW(NameFormat, None, nSize)
    except OSError as e:
        if e.winerror != ERROR_MORE_DATA:
            raise
    if not nSize.contents.value:
        return None
    lpNameBuffer = ctypes.create_unicode_buffer(nSize.contents.value)
    secur32.GetUserNameExW(NameFormat, lpNameBuffer, nSize)
    return lpNameBuffer.value