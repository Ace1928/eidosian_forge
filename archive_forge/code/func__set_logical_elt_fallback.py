import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def _set_logical_elt_fallback(vec, i: int, value):
    LOGICAL(vec)[i] = value