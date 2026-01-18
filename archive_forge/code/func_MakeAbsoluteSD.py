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
def MakeAbsoluteSD(pSelfRelativeSecurityDescriptor: Any) -> Any:
    pAbsoluteSecurityDescriptor = None
    lpdwAbsoluteSecurityDescriptorSize = wintypes.DWORD(0)
    pDacl = None
    lpdwDaclSize = wintypes.DWORD(0)
    pSacl = None
    lpdwSaclSize = wintypes.DWORD(0)
    pOwner = None
    lpdwOwnerSize = wintypes.DWORD(0)
    pPrimaryGroup = None
    lpdwPrimaryGroupSize = wintypes.DWORD(0)
    try:
        advapi32.MakeAbsoluteSD(pSelfRelativeSecurityDescriptor, pAbsoluteSecurityDescriptor, ctypes.byref(lpdwAbsoluteSecurityDescriptorSize), pDacl, ctypes.byref(lpdwDaclSize), pSacl, ctypes.byref(lpdwSaclSize), pOwner, ctypes.byref(lpdwOwnerSize), pPrimaryGroup, ctypes.byref(lpdwPrimaryGroupSize))
    except OSError as e:
        if e.winerror != ERROR_INSUFFICIENT_BUFFER:
            raise
    pAbsoluteSecurityDescriptor = (wintypes.BYTE * lpdwAbsoluteSecurityDescriptorSize.value)()
    pDaclData = (wintypes.BYTE * lpdwDaclSize.value)()
    pDacl = ctypes.cast(pDaclData, PACL).contents
    pSaclData = (wintypes.BYTE * lpdwSaclSize.value)()
    pSacl = ctypes.cast(pSaclData, PACL).contents
    pOwnerData = (wintypes.BYTE * lpdwOwnerSize.value)()
    pOwner = ctypes.cast(pOwnerData, PSID)
    pPrimaryGroupData = (wintypes.BYTE * lpdwPrimaryGroupSize.value)()
    pPrimaryGroup = ctypes.cast(pPrimaryGroupData, PSID)
    advapi32.MakeAbsoluteSD(pSelfRelativeSecurityDescriptor, pAbsoluteSecurityDescriptor, ctypes.byref(lpdwAbsoluteSecurityDescriptorSize), pDacl, ctypes.byref(lpdwDaclSize), pSacl, ctypes.byref(lpdwSaclSize), pOwner, lpdwOwnerSize, pPrimaryGroup, ctypes.byref(lpdwPrimaryGroupSize))
    return pAbsoluteSecurityDescriptor