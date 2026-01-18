from __future__ import annotations
from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import contextlib
from dataclasses import dataclass, field
import functools
from .compat import io
import itertools
import os
import re
import sys
from typing import Iterable
@dataclass
class _ReadState:
    elements_added: set[str] = field(default_factory=set)
    cursect: dict[str, str] | None = None
    sectname: str | None = None
    optname: str | None = None
    lineno: int = 0
    indent_level: int = 0
    errors: list[ParsingError] = field(default_factory=list)