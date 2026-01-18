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
def _evaluated_promise(function):

    def _(*args, **kwargs):
        robj = function(*args, **kwargs)
        if _TYPEOF(robj) == openrlib.rlib.PROMSXP:
            robj = openrlib.rlib.Rf_eval(robj, openrlib.rlib.R_GlobalEnv)
        return robj
    return _