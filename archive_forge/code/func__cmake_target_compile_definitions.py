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
def _cmake_target_compile_definitions(self, tline: CMakeTraceLine) -> None:
    self._parse_common_target_options('target_compile_definitions', 'COMPILE_DEFINITIONS', 'INTERFACE_COMPILE_DEFINITIONS', tline)