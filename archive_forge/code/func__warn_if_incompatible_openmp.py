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
def _warn_if_incompatible_openmp(self):
    """Raise a warning if llvm-OpenMP and intel-OpenMP are both loaded"""
    prefixes = [lib_controller.prefix for lib_controller in self.lib_controllers]
    msg = textwrap.dedent("\n            Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at\n            the same time. Both libraries are known to be incompatible and this\n            can cause random crashes or deadlocks on Linux when loaded in the\n            same Python program.\n            Using threadpoolctl may cause crashes or deadlocks. For more\n            information and possible workarounds, please see\n                https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md\n            ")
    if 'libomp' in prefixes and 'libiomp' in prefixes:
        warnings.warn(msg, RuntimeWarning)