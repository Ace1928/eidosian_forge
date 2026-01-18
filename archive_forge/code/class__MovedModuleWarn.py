from __future__ import annotations
import importlib
import sys
import types
import typing
import warnings
from urwid.canvas import (
from urwid.command_map import (
from urwid.font import (
from urwid.signals import (
from urwid.str_util import calc_text_pos, calc_width, is_wide_char, move_next_char, move_prev_char, within_double_byte
from urwid.text_layout import LayoutSegment, StandardTextLayout, TextLayout, default_layout
from urwid.util import (
from urwid.version import version as __version__
from urwid.version import version_tuple as __version_tuple__
from . import display, event_loop, widget
from .display import (
from .event_loop import AsyncioEventLoop, EventLoop, ExitMainLoop, MainLoop, SelectEventLoop
from .widget import (
class _MovedModuleWarn(_MovedModule):
    """Special class to handle moved modules.

    Produce DeprecationWarning messages for imports.
    """
    __slots__ = ()

    def __getattr__(self, name: str) -> typing.Any:
        warnings.warn(f'{self._moved_from} is moved to {self._moved_to}', DeprecationWarning, stacklevel=2)
        return super().__getattr__(name)