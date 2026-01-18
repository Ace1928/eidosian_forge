import inspect
import os
import platform
import sys
import threading
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from getpass import getpass
from html import escape
from inspect import isclass
from itertools import islice
from math import ceil
from time import monotonic
from types import FrameType, ModuleType, TracebackType
from typing import (
from pip._vendor.rich._null_file import NULL_FILE
from . import errors, themes
from ._emoji_replace import _emoji_replace
from ._export_format import CONSOLE_HTML_FORMAT, CONSOLE_SVG_FORMAT
from ._fileno import get_fileno
from ._log_render import FormatTimeCallable, LogRender
from .align import Align, AlignMethod
from .color import ColorSystem, blend_rgb
from .control import Control
from .emoji import EmojiVariant
from .highlighter import NullHighlighter, ReprHighlighter
from .markup import render as render_markup
from .measure import Measurement, measure_renderables
from .pager import Pager, SystemPager
from .pretty import Pretty, is_expandable
from .protocol import rich_cast
from .region import Region
from .scope import render_scope
from .screen import Screen
from .segment import Segment
from .style import Style, StyleType
from .styled import Styled
from .terminal_theme import DEFAULT_TERMINAL_THEME, SVG_EXPORT_THEME, TerminalTheme
from .text import Text, TextType
from .theme import Theme, ThemeStack
def _collect_renderables(self, objects: Iterable[Any], sep: str, end: str, *, justify: Optional[JustifyMethod]=None, emoji: Optional[bool]=None, markup: Optional[bool]=None, highlight: Optional[bool]=None) -> List[ConsoleRenderable]:
    """Combine a number of renderables and text into one renderable.

        Args:
            objects (Iterable[Any]): Anything that Rich can render.
            sep (str): String to write between print data.
            end (str): String to write at end of print data.
            justify (str, optional): One of "left", "right", "center", or "full". Defaults to ``None``.
            emoji (Optional[bool], optional): Enable emoji code, or ``None`` to use console default.
            markup (Optional[bool], optional): Enable markup, or ``None`` to use console default.
            highlight (Optional[bool], optional): Enable automatic highlighting, or ``None`` to use console default.

        Returns:
            List[ConsoleRenderable]: A list of things to render.
        """
    renderables: List[ConsoleRenderable] = []
    _append = renderables.append
    text: List[Text] = []
    append_text = text.append
    append = _append
    if justify in ('left', 'center', 'right'):

        def align_append(renderable: RenderableType) -> None:
            _append(Align(renderable, cast(AlignMethod, justify)))
        append = align_append
    _highlighter: HighlighterType = _null_highlighter
    if highlight or (highlight is None and self._highlight):
        _highlighter = self.highlighter

    def check_text() -> None:
        if text:
            sep_text = Text(sep, justify=justify, end=end)
            append(sep_text.join(text))
            text.clear()
    for renderable in objects:
        renderable = rich_cast(renderable)
        if isinstance(renderable, str):
            append_text(self.render_str(renderable, emoji=emoji, markup=markup, highlighter=_highlighter))
        elif isinstance(renderable, Text):
            append_text(renderable)
        elif isinstance(renderable, ConsoleRenderable):
            check_text()
            append(renderable)
        elif is_expandable(renderable):
            check_text()
            append(Pretty(renderable, highlighter=_highlighter))
        else:
            append_text(_highlighter(str(renderable)))
    check_text()
    if self.style is not None:
        style = self.get_style(self.style)
        renderables = [Styled(renderable, style) for renderable in renderables]
    return renderables