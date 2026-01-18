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
def _lex_trace_json(self, trace: str) -> T.Generator[CMakeTraceLine, None, None]:
    lines = trace.splitlines(keepends=False)
    lines.pop(0)
    for i in lines:
        data = json.loads(i)
        assert isinstance(data['file'], str)
        assert isinstance(data['line'], int)
        assert isinstance(data['cmd'], str)
        assert isinstance(data['args'], list)
        args = data['args']
        for j in args:
            assert isinstance(j, str)
        yield CMakeTraceLine(data['file'], data['line'], data['cmd'], args)