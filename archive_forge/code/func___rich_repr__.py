from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas, SolidCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError, WidgetWarning
def __rich_repr__(self) -> Iterator[tuple[str | None, typing.Any] | typing.Any]:
    yield ('w', self.original_widget)
    yield ('align', self.align)
    yield ('width', self.width)
    yield ('min_width', self.min_width)
    yield ('left', self.left)
    yield ('right', self.right)