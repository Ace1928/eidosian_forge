from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
def _merge_attrs(list_of_attrs: list[Attrs]) -> Attrs:
    """
    Take a list of :class:`.Attrs` instances and merge them into one.
    Every `Attr` in the list can override the styling of the previous one. So,
    the last one has highest priority.
    """

    def _or(*values: _T) -> _T:
        """Take first not-None value, starting at the end."""
        for v in values[::-1]:
            if v is not None:
                return v
        raise ValueError
    return Attrs(color=_or('', *[a.color for a in list_of_attrs]), bgcolor=_or('', *[a.bgcolor for a in list_of_attrs]), bold=_or(False, *[a.bold for a in list_of_attrs]), underline=_or(False, *[a.underline for a in list_of_attrs]), strike=_or(False, *[a.strike for a in list_of_attrs]), italic=_or(False, *[a.italic for a in list_of_attrs]), blink=_or(False, *[a.blink for a in list_of_attrs]), reverse=_or(False, *[a.reverse for a in list_of_attrs]), hidden=_or(False, *[a.hidden for a in list_of_attrs]))