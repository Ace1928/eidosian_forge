from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._processevents_def, openrlib._rinterface_cffi)
def _processevents() -> None:
    try:
        processevents()
    except KeyboardInterrupt:
        rlib = openrlib.rlib
        if os.name == 'nt':
            rlib.UserBreak = 1
        else:
            rlib.R_interrupts_pending = 1
    except Exception as e:
        logger.error(_PROCESSEVENTS_EXCEPTION_LOG, str(e))