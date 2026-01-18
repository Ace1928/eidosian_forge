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
def _geterrmessage() -> str:
    rlib = openrlib.rlib
    with memorymanagement.rmemory() as rmemory:
        symbol = rmemory.protect(rlib.Rf_install(conversion._str_to_cchar('geterrmessage')))
        geterrmessage = _findvar(symbol, rlib.R_GlobalEnv)
        call_r = rlib.Rf_lang1(geterrmessage)
        res = rmemory.protect(rlib.Rf_eval(call_r, rlib.R_GlobalEnv))
        res = _string_getitem(res, 0)
    return res