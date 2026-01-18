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
def _meson_ps_reload_vars(self, tline: CMakeTraceLine) -> None:
    self.delayed_commands = self.get_cmake_var('MESON_PS_DELAYED_CALLS')