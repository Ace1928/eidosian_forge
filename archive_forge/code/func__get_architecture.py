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
def _get_architecture(self):
    """Return the architecture detected by BLIS"""
    bli_arch_query_id = getattr(self.dynlib, 'bli_arch_query_id', None)
    bli_arch_string = getattr(self.dynlib, 'bli_arch_string', None)
    if bli_arch_query_id is None or bli_arch_string is None:
        return None
    bli_arch_query_id.restype = ctypes.c_int
    bli_arch_string.restype = ctypes.c_char_p
    return bli_arch_string(bli_arch_query_id()).decode('utf-8')