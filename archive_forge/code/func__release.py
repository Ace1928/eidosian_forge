import abc
import enum
import inspect
import logging
from typing import Tuple
import typing
import warnings
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import conversion
from rpy2.rinterface_lib import embedded
from rpy2.rinterface_lib import memorymanagement
from _cffi_backend import FFI  # type: ignore
def _release(cdata: FFI.CData) -> None:
    addr = int(ffi.cast('uintptr_t', cdata))
    count = _R_PRESERVED[addr] - 1
    if count == 0:
        del _R_PRESERVED[addr]
        openrlib.rlib.R_ReleaseObject(cdata)
    else:
        _R_PRESERVED[addr] = count