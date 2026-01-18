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
class ThreadedStream:
    """Thread-local wrapper for sys.stdout for the interactive console."""

    @staticmethod
    def push() -> None:
        if not isinstance(sys.stdout, ThreadedStream):
            sys.stdout = t.cast(t.TextIO, ThreadedStream())
        _stream.set(HTMLStringO())

    @staticmethod
    def fetch() -> str:
        try:
            stream = _stream.get()
        except LookupError:
            return ''
        return stream.reset()

    @staticmethod
    def displayhook(obj: object) -> None:
        try:
            stream = _stream.get()
        except LookupError:
            return _displayhook(obj)
        if obj is not None:
            _ipy.get().locals['_'] = obj
            stream._write(debug_repr(obj))

    def __setattr__(self, name: str, value: t.Any) -> None:
        raise AttributeError(f'read only attribute {name}')

    def __dir__(self) -> list[str]:
        return dir(sys.__stdout__)

    def __getattribute__(self, name: str) -> t.Any:
        try:
            stream = _stream.get()
        except LookupError:
            stream = sys.__stdout__
        return getattr(stream, name)

    def __repr__(self) -> str:
        return repr(sys.__stdout__)