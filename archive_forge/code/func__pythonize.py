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
def _pythonize(value: str) -> None | bool | int | float | str:
    if value in _PYTHON_CONSTANTS:
        return _PYTHON_CONSTANTS[value]
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            pass
    if value[:1] == value[-1:] and value[0] in '"\'':
        value = value[1:-1]
    return str(value)