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
def _cmake_add_dependencies(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    if len(args) < 2:
        return self._gen_exception('add_dependencies', 'takes at least 2 arguments', tline)
    target = self.targets.get(args[0])
    if not target:
        return self._gen_exception('add_dependencies', 'target not found', tline)
    for i in args[1:]:
        target.depends += i.split(';')