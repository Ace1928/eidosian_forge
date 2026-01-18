from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
class WhateverIO(StringIO):
    """StringIO that takes bytes or str."""

    def __init__(self, v: bytes | str | None=None, *a: Any, **kw: Any) -> None:
        _SIO_init(self, v.decode() if isinstance(v, bytes) else v, *a, **kw)

    def write(self, data: bytes | str) -> int:
        return _SIO_write(self, data.decode() if isinstance(data, bytes) else data)