from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._consoleflush_def, openrlib._rinterface_cffi)
def _consoleflush() -> None:
    try:
        consoleflush()
    except Exception as e:
        logger.error(_FLUSHCONSOLE_EXCEPTION_LOG, str(e))