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
def _cmake_add_executable(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    is_imported = True
    if 'IMPORTED' not in args:
        return self._gen_exception('add_executable', 'non imported executables are not supported', tline)
    args.remove('IMPORTED')
    if len(args) < 1:
        return self._gen_exception('add_executable', 'requires at least 1 argument', tline)
    self.targets[args[0]] = CMakeTarget(args[0], 'EXECUTABLE', {}, tline=tline, imported=is_imported)