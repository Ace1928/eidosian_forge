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
def _get_current_backend(self):
    """Return the backend of FlexiBLAS"""
    get_backend_ = getattr(self.dynlib, 'flexiblas_current_backend', None)
    if get_backend_ is None:
        return None
    backend = ctypes.create_string_buffer(1024)
    get_backend_(backend, ctypes.sizeof(backend))
    return backend.value.decode('utf-8')