from __future__ import annotations
import os
import tempfile
import textwrap
from functools import lru_cache
class CppWrapperCodeGenError(RuntimeError):

    def __init__(self, msg: str):
        super().__init__(f'C++ wrapper codegen error: {msg}')