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
def _cmake_unset(self, tline: CMakeTraceLine) -> None:
    if len(tline.args) < 1:
        return self._gen_exception('unset', 'requires at least one argument', tline)
    if tline.args[0] in self.vars:
        del self.vars[tline.args[0]]