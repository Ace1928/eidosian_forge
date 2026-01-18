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
def _check_widget_subclass(widget: Widget | None) -> None:
    if widget is None:
        return
    if not isinstance(widget, Widget):
        obj_class_path = f'{widget.__class__.__module__}.{widget.__class__.__name__}'
        warnings.warn(f'{obj_class_path} is not subclass of Widget', DeprecationWarning, stacklevel=3)