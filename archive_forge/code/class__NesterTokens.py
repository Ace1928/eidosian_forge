from __future__ import annotations
from collections.abc import Generator, Sequence
import textwrap
from typing import Any, NamedTuple, TypeVar, overload
from .token import Token
class _NesterTokens(NamedTuple):
    opening: Token
    closing: Token