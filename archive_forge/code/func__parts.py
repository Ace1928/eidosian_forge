from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
def _parts(ops: list[tuple[bool, str]]) -> list[ast.AST]:
    parts = [_convert(elem) if is_dynamic else ast.Constant(elem) for is_dynamic, elem in ops]
    parts = parts or [ast.Constant('')]
    ret = [parts[0]]
    for p in parts[1:]:
        if isinstance(p, ast.Constant) and isinstance(ret[-1], ast.Constant):
            ret[-1] = ast.Constant(ret[-1].value + p.value)
        else:
            ret.append(p)
    return ret