from __future__ import annotations
import functools
import logging
import typing
import warnings
from operator import attrgetter
from urwid import signals
from urwid.canvas import Canvas, CanvasCache, CompositeCanvas
from urwid.command_map import command_map
from urwid.split_repr import split_repr
from urwid.util import MetaSuper
from .constants import Sizing
def _set_w(self, w: WrappedWidget) -> None:
    """
        Change the wrapped widget.  This is meant to be called
        only by subclasses.
        >>> from urwid import Edit, Text
        >>> size = (10,)
        >>> ww = WidgetWrap(Edit("hello? ","hi"))
        >>> ww.render(size).text # ... = b in Python 3
        [...'hello? hi ']
        >>> ww.selectable()
        True
        >>> ww._w = Text("goodbye") # calls _set_w()
        >>> ww.render(size).text
        [...'goodbye   ']
        >>> ww.selectable()
        False
        """
    warnings.warn("_set_w is deprecated. Please use 'WidgetWrap._w' property directly", DeprecationWarning, stacklevel=2)
    self._wrapped_widget = w
    self._invalidate()