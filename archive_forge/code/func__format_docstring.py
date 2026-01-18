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
def _format_docstring(*args, **kwargs):

    def decorator(o):
        if o.__doc__ is not None:
            o.__doc__ = o.__doc__.format(*args, **kwargs)
        return o
    return decorator