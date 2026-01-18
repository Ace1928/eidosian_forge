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
def CreateWellKnownSid(WellKnownSidType: Any) -> Any:
    pSid = (ctypes.c_char * 1)()
    cbSid = wintypes.DWORD()
    try:
        advapi32.CreateWellKnownSid(WellKnownSidType, None, pSid, ctypes.byref(cbSid))
    except OSError as e:
        if e.winerror != ERROR_INSUFFICIENT_BUFFER:
            raise
        pSid = (ctypes.c_char * cbSid.value)()
        advapi32.CreateWellKnownSid(WellKnownSidType, None, pSid, ctypes.byref(cbSid))
    return pSid[:]