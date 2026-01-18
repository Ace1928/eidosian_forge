from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError
Pass to self.original_widget.