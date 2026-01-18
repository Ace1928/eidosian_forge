from __future__ import annotations
from asyncio import FIRST_COMPLETED, Future, ensure_future, sleep, wait
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Hashable
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Char, Screen, WritePosition
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.styles import (
class _StyleStringToAttrsCache(Dict[str, Attrs]):
    """
    A cache structure that maps style strings to :class:`.Attr`.
    (This is an important speed up.)
    """

    def __init__(self, get_attrs_for_style_str: Callable[[str], Attrs], style_transformation: StyleTransformation) -> None:
        self.get_attrs_for_style_str = get_attrs_for_style_str
        self.style_transformation = style_transformation

    def __missing__(self, style_str: str) -> Attrs:
        attrs = self.get_attrs_for_style_str(style_str)
        attrs = self.style_transformation.transform_attrs(attrs)
        self[style_str] = attrs
        return attrs