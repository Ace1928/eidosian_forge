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
def _load_libraries(self):
    """Loop through loaded shared libraries and store the supported ones"""
    if sys.platform == 'darwin':
        self._find_libraries_with_dyld()
    elif sys.platform == 'win32':
        self._find_libraries_with_enum_process_module_ex()
    else:
        self._find_libraries_with_dl_iterate_phdr()