from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas, SolidCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError, WidgetWarning
def _repr_attrs(self) -> dict[str, typing.Any]:
    attrs = {**super()._repr_attrs(), 'align': self.align, 'width': self.width, 'left': self.left, 'right': self.right, 'min_width': self.min_width}
    return remove_defaults(attrs, Padding.__init__)