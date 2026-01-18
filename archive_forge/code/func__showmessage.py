from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
@ffi_proxy.callback(ffi_proxy._showmessage_def, openrlib._rinterface_cffi)
def _showmessage(buf):
    s = conversion._cchar_to_str(buf, _CCHAR_ENCODING)
    try:
        showmessage(s)
    except Exception as e:
        logger.error(_SHOWMESSAGE_EXCEPTION_LOG, str(e))