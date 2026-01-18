from __future__ import annotations
import code
import sys
import typing as t
from contextvars import ContextVar
from types import CodeType
from markupsafe import escape
from .repr import debug_repr
from .repr import dump
from .repr import helper
class HTMLStringO:
    """A StringO version that HTML escapes on write."""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def seek(self, n: int, mode: int=0) -> None:
        pass

    def readline(self) -> str:
        if len(self._buffer) == 0:
            return ''
        ret = self._buffer[0]
        del self._buffer[0]
        return ret

    def reset(self) -> str:
        val = ''.join(self._buffer)
        del self._buffer[:]
        return val

    def _write(self, x: str) -> None:
        self._buffer.append(x)

    def write(self, x: str) -> None:
        self._write(escape(x))

    def writelines(self, x: t.Iterable[str]) -> None:
        self._write(escape(''.join(x)))