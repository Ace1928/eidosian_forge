from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class InvalidCxxCompiler(RuntimeError):

    def __init__(self):
        from . import config
        super().__init__(f'No working C++ compiler found in {config.__name__}.cpp.cxx: {config.cpp.cxx}')