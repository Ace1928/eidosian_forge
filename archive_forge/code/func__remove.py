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
def _remove(name: FFI.CData, env: FFI.CData, inherits) -> FFI.CData:
    rlib = openrlib.rlib
    with memorymanagement.rmemory() as rmemory:
        internal = rmemory.protect(rlib.Rf_install(conversion._str_to_cchar('.Internal')))
        remove = rmemory.protect(rlib.Rf_install(conversion._str_to_cchar('remove')))
        args = rmemory.protect(rlib.Rf_lang4(remove, name, env, inherits))
        call = rmemory.protect(rlib.Rf_lang2(internal, args))
        res = rlib.Rf_eval(call, rlib.R_GlobalEnv)
    return res