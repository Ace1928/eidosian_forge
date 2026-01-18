from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _gen_exception(self, function: str, error: str, tline: CMakeTraceLine) -> None:
    if self.permissive:
        mlog.debug(f'CMake trace warning: {function}() {error}\n{tline}')
        return None
    raise CMakeException(f'CMake: {function}() {error}\n{tline}')