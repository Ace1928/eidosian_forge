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
def __del__(self):
    addr = get_rid(self._cdata)
    _release(self._cdata)
    if addr not in _PY_PASSENGER:
        del _PY_PASSENGER[addr]