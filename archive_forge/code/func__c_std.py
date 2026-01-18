import ctypes
import logging
import os
import sys
from contextlib import contextmanager
from functools import partial
def _c_std(stream: str):
    return ctypes.c_void_p.in_dll(libc, stream)