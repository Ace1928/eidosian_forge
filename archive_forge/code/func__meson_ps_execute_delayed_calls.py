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
def _meson_ps_execute_delayed_calls(self, tline: CMakeTraceLine) -> None:
    for l in self.stored_commands:
        fn = self.functions.get(l.func, None)
        if fn:
            fn(l)
    self.stored_commands = []