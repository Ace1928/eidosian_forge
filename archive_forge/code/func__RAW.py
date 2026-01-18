import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _RAW(x):
    return ffi.cast('Rbyte *', DATAPTR(x))