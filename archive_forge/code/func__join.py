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
def _join(parts: list[ast.AST]) -> ast.AST:
    if len(parts) == 1:
        return parts[0]
    return ast.JoinedStr(parts)