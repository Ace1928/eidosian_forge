from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas
from .widget import Widget, WidgetError, WidgetWarning, delegate_to_widget_mixin
class WidgetPlaceholder(delegate_to_widget_mixin('_original_widget'), WidgetDecoration[WrappedWidget]):
    """
    This is a do-nothing decoration widget that can be used for swapping
    between widgets without modifying the container of this widget.

    This can be useful for making an interface with a number of distinct
    pages or for showing and hiding menu or status bars.

    The widget displayed is stored as the self.original_widget property and
    can be changed by assigning a new widget to it.
    """