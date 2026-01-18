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
class PagerContext:
    """A context manager that 'pages' content. See :meth:`~rich.console.Console.pager` for usage."""

    def __init__(self, console: 'Console', pager: Optional[Pager]=None, styles: bool=False, links: bool=False) -> None:
        self._console = console
        self.pager = SystemPager() if pager is None else pager
        self.styles = styles
        self.links = links

    def __enter__(self) -> 'PagerContext':
        self._console._enter_buffer()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if exc_type is None:
            with self._console._lock:
                buffer: List[Segment] = self._console._buffer[:]
                del self._console._buffer[:]
                segments: Iterable[Segment] = buffer
                if not self.styles:
                    segments = Segment.strip_styles(segments)
                elif not self.links:
                    segments = Segment.strip_links(segments)
                content = self._console._render_buffer(segments)
            self.pager.show(content)
        self._console._exit_buffer()