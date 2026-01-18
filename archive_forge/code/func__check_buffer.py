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
def _check_buffer(self) -> None:
    """Check if the buffer may be rendered. Render it if it can (e.g. Console.quiet is False)
        Rendering is supported on Windows, Unix and Jupyter environments. For
        legacy Windows consoles, the win32 API is called directly.
        This method will also record what it renders if recording is enabled via Console.record.
        """
    if self.quiet:
        del self._buffer[:]
        return
    with self._lock:
        if self.record:
            with self._record_buffer_lock:
                self._record_buffer.extend(self._buffer[:])
        if self._buffer_index == 0:
            if self.is_jupyter:
                from .jupyter import display
                display(self._buffer, self._render_buffer(self._buffer[:]))
                del self._buffer[:]
            else:
                if WINDOWS:
                    use_legacy_windows_render = False
                    if self.legacy_windows:
                        fileno = get_fileno(self.file)
                        if fileno is not None:
                            use_legacy_windows_render = fileno in _STD_STREAMS_OUTPUT
                    if use_legacy_windows_render:
                        from pip._vendor.rich._win32_console import LegacyWindowsTerm
                        from pip._vendor.rich._windows_renderer import legacy_windows_render
                        buffer = self._buffer[:]
                        if self.no_color and self._color_system:
                            buffer = list(Segment.remove_color(buffer))
                        legacy_windows_render(buffer, LegacyWindowsTerm(self.file))
                    else:
                        text = self._render_buffer(self._buffer[:])
                        write = self.file.write
                        MAX_WRITE = 32 * 1024 // 4
                        try:
                            if len(text) <= MAX_WRITE:
                                write(text)
                            else:
                                batch: List[str] = []
                                batch_append = batch.append
                                size = 0
                                for line in text.splitlines(True):
                                    if size + len(line) > MAX_WRITE and batch:
                                        write(''.join(batch))
                                        batch.clear()
                                        size = 0
                                    batch_append(line)
                                    size += len(line)
                                if batch:
                                    write(''.join(batch))
                                    batch.clear()
                        except UnicodeEncodeError as error:
                            error.reason = f'{error.reason}\n*** You may need to add PYTHONIOENCODING=utf-8 to your environment ***'
                            raise
                else:
                    text = self._render_buffer(self._buffer[:])
                    try:
                        self.file.write(text)
                    except UnicodeEncodeError as error:
                        error.reason = f'{error.reason}\n*** You may need to add PYTHONIOENCODING=utf-8 to your environment ***'
                        raise
                self.file.flush()
                del self._buffer[:]