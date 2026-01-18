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
@dataclass
class ConsoleOptions:
    """Options for __rich_console__ method."""
    size: ConsoleDimensions
    'Size of console.'
    legacy_windows: bool
    'legacy_windows: flag for legacy windows.'
    min_width: int
    'Minimum width of renderable.'
    max_width: int
    'Maximum width of renderable.'
    is_terminal: bool
    'True if the target is a terminal, otherwise False.'
    encoding: str
    'Encoding of terminal.'
    max_height: int
    'Height of container (starts as terminal)'
    justify: Optional[JustifyMethod] = None
    'Justify value override for renderable.'
    overflow: Optional[OverflowMethod] = None
    'Overflow value override for renderable.'
    no_wrap: Optional[bool] = False
    'Disable wrapping for text.'
    highlight: Optional[bool] = None
    'Highlight override for render_str.'
    markup: Optional[bool] = None
    'Enable markup when rendering strings.'
    height: Optional[int] = None

    @property
    def ascii_only(self) -> bool:
        """Check if renderables should use ascii only."""
        return not self.encoding.startswith('utf')

    def copy(self) -> 'ConsoleOptions':
        """Return a copy of the options.

        Returns:
            ConsoleOptions: a copy of self.
        """
        options: ConsoleOptions = ConsoleOptions.__new__(ConsoleOptions)
        options.__dict__ = self.__dict__.copy()
        return options

    def update(self, *, width: Union[int, NoChange]=NO_CHANGE, min_width: Union[int, NoChange]=NO_CHANGE, max_width: Union[int, NoChange]=NO_CHANGE, justify: Union[Optional[JustifyMethod], NoChange]=NO_CHANGE, overflow: Union[Optional[OverflowMethod], NoChange]=NO_CHANGE, no_wrap: Union[Optional[bool], NoChange]=NO_CHANGE, highlight: Union[Optional[bool], NoChange]=NO_CHANGE, markup: Union[Optional[bool], NoChange]=NO_CHANGE, height: Union[Optional[int], NoChange]=NO_CHANGE) -> 'ConsoleOptions':
        """Update values, return a copy."""
        options = self.copy()
        if not isinstance(width, NoChange):
            options.min_width = options.max_width = max(0, width)
        if not isinstance(min_width, NoChange):
            options.min_width = min_width
        if not isinstance(max_width, NoChange):
            options.max_width = max_width
        if not isinstance(justify, NoChange):
            options.justify = justify
        if not isinstance(overflow, NoChange):
            options.overflow = overflow
        if not isinstance(no_wrap, NoChange):
            options.no_wrap = no_wrap
        if not isinstance(highlight, NoChange):
            options.highlight = highlight
        if not isinstance(markup, NoChange):
            options.markup = markup
        if not isinstance(height, NoChange):
            if height is not None:
                options.max_height = height
            options.height = None if height is None else max(0, height)
        return options

    def update_width(self, width: int) -> 'ConsoleOptions':
        """Update just the width, return a copy.

        Args:
            width (int): New width (sets both min_width and max_width)

        Returns:
            ~ConsoleOptions: New console options instance.
        """
        options = self.copy()
        options.min_width = options.max_width = max(0, width)
        return options

    def update_height(self, height: int) -> 'ConsoleOptions':
        """Update the height, and return a copy.

        Args:
            height (int): New height

        Returns:
            ~ConsoleOptions: New Console options instance.
        """
        options = self.copy()
        options.max_height = options.height = height
        return options

    def reset_height(self) -> 'ConsoleOptions':
        """Return a copy of the options with height set to ``None``.

        Returns:
            ~ConsoleOptions: New console options instance.
        """
        options = self.copy()
        options.height = None
        return options

    def update_dimensions(self, width: int, height: int) -> 'ConsoleOptions':
        """Update the width and height, and return a copy.

        Args:
            width (int): New width (sets both min_width and max_width).
            height (int): New height.

        Returns:
            ~ConsoleOptions: New console options instance.
        """
        options = self.copy()
        options.min_width = options.max_width = max(0, width)
        options.height = options.max_height = height
        return options