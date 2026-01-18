from __future__ import annotations
from pathlib import Path
from .traceparser import CMakeTraceParser
from ..envconfig import CMakeSkipCompilerTest
from .common import language_map, cmake_get_generator_args
from .. import mlog
import shutil
import typing as T
from enum import Enum
from textwrap import dedent
@staticmethod
def _print_vars(vars: T.Dict[str, T.List[str]]) -> str:
    res = ''
    for key, value in vars.items():
        res += 'set(' + key
        for i in value:
            res += f' "{i}"'
        res += ')\n'
    return res