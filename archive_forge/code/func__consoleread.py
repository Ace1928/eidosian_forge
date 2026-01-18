from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._consoleread_def, openrlib._rinterface_cffi)
def _consoleread(prompt, buf, n: int, addtohistory) -> int:
    success = None
    try:
        s = conversion._cchar_to_str(prompt, _CCHAR_ENCODING)
        reply = consoleread(s)
    except Exception as e:
        success = 0
        logger.error(_READCONSOLE_EXCEPTION_LOG, str(e))
    if success == 0:
        return success
    try:
        reply_b = reply.encode('utf-8')
        reply_n = min(n, len(reply_b))
        pybuf = bytearray(n)
        pybuf[:reply_n] = reply_b[:reply_n]
        pybuf[reply_n] = ord('\n')
        pybuf[reply_n + 1] = 0
        openrlib.ffi.memmove(buf, pybuf, n)
        if reply_n == 0:
            success = 0
        else:
            success = 1
    except Exception as e:
        success = 0
        logger.error(_READCONSOLE_INTERNAL_EXCEPTION_LOG, str(e))
    return success