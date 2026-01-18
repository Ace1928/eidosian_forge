import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _VECTOR_ELT(robj, i):
    return ffi.cast('SEXP *', DATAPTR(robj))[i]