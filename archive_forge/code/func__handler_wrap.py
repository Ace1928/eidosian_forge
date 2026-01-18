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
@ffi_proxy.callback(ffi_proxy._handler_def, openrlib._rinterface_cffi)
def _handler_wrap(cond, hdata):
    return openrlib.rlib.R_NilValue