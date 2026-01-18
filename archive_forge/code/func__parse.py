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
def _parse(cdata: FFI.CData, num, rmemory) -> FFI.CData:
    status = ffi.new('ParseStatus[1]', None)
    data = ffi.new_handle((cdata, num, status))
    hdata = ffi.NULL
    res = rmemory.protect(openrlib.rlib.R_tryCatchError(_parsevector_wrap, data, _handler_wrap, hdata))
    if status[0] != openrlib.rlib.PARSE_OK:
        raise RParsingError('Parsing status not OK', status=PARSING_STATUS(status[0]))
    return res