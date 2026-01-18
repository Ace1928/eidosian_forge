from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasCombine, CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import is_mouse_press
from .constants import Sizing, VAlign
from .container import WidgetContainerMixin
from .filler import Filler
from .widget import Widget, WidgetError
def _contents_keys(self) -> list[Literal['header', 'footer', 'body']]:
    keys = ['body']
    if self._header:
        keys.append('header')
    if self._footer:
        keys.append('footer')
    return keys