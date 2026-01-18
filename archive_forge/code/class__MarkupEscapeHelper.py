import functools
import re
import string
import sys
import typing as t
class _MarkupEscapeHelper:
    """Helper for :meth:`Markup.__mod__`."""
    __slots__ = ('obj', 'escape')

    def __init__(self, obj: t.Any, escape: t.Callable[[t.Any], Markup]) -> None:
        self.obj = obj
        self.escape = escape

    def __getitem__(self, item: t.Any) -> 'te.Self':
        return self.__class__(self.obj[item], self.escape)

    def __str__(self) -> str:
        return str(self.escape(self.obj))

    def __repr__(self) -> str:
        return str(self.escape(repr(self.obj)))

    def __int__(self) -> int:
        return int(self.obj)

    def __float__(self) -> float:
        return float(self.obj)