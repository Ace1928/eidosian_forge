from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._consolereset_def, openrlib._rinterface_cffi)
def _consolereset() -> None:
    try:
        consolereset()
    except Exception as e:
        logger.error(_RESETCONSOLE_EXCEPTION_LOG, str(e))