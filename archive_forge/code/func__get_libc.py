import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
@classmethod
def _get_libc(cls):
    """Load the lib-C for unix systems."""
    libc = cls._system_libraries.get('libc')
    if libc is None:
        libc_name = find_library('c')
        if libc_name is None:
            warnings.warn(f'libc not found. The ctypes module in Python {sys.version_info.major}.{sys.version_info.minor} is maybe too old for this OS.', RuntimeWarning)
            return None
        libc = ctypes.CDLL(libc_name, mode=_RTLD_NOLOAD)
        cls._system_libraries['libc'] = libc
    return libc