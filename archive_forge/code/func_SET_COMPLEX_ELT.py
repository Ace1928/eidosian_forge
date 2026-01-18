import logging
import os
import platform
import threading
import typing
import rpy2.situation
from rpy2.rinterface_lib import ffi_proxy
def SET_COMPLEX_ELT(vec, i: int, value: complex):
    COMPLEX(vec)[i].r = value.real
    COMPLEX(vec)[i].i = value.imag