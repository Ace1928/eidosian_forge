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
def _check_prefix(self, library_basename, filename_prefixes):
    """Return the prefix library_basename starts with

        Return None if none matches.
        """
    for prefix in filename_prefixes:
        if library_basename.startswith(prefix):
            return prefix
    return None