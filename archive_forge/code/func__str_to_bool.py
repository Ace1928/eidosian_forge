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
def _str_to_bool(self, expr: T.Union[str, T.List[str]]) -> bool:
    if not expr:
        return False
    if isinstance(expr, list):
        expr_str = expr[0]
    else:
        expr_str = expr
    expr_str = expr_str.upper()
    return expr_str not in ['0', 'OFF', 'NO', 'FALSE', 'N', 'IGNORE'] and (not expr_str.endswith('NOTFOUND'))