from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class CUDACompileError(CppCompileError):
    pass