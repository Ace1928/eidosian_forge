import sys
from functools import lru_cache
from marshal import dumps, loads
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast
from . import errors
from .color import Color, ColorParseError, ColorSystem, blend_rgb
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME, TerminalTheme
class StyleStack:
    """A stack of styles."""
    __slots__ = ['_stack']

    def __init__(self, default_style: 'Style') -> None:
        self._stack: List[Style] = [default_style]

    def __repr__(self) -> str:
        return f'<stylestack {self._stack!r}>'

    @property
    def current(self) -> Style:
        """Get the Style at the top of the stack."""
        return self._stack[-1]

    def push(self, style: Style) -> None:
        """Push a new style on to the stack.

        Args:
            style (Style): New style to combine with current style.
        """
        self._stack.append(self._stack[-1] + style)

    def pop(self) -> Style:
        """Pop last style and discard.

        Returns:
            Style: New current style (also available as stack.current)
        """
        self._stack.pop()
        return self._stack[-1]