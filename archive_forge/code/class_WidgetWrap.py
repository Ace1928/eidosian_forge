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
class WidgetWrap(delegate_to_widget_mixin('_wrapped_widget'), typing.Generic[WrappedWidget]):

    def __init__(self, w: WrappedWidget) -> None:
        """
        w -- widget to wrap, stored as self._w

        This object will pass the functions defined in Widget interface
        definition to self._w.

        The purpose of this widget is to provide a base class for
        widgets that compose other widgets for their display and
        behaviour.  The details of that composition should not affect
        users of the subclass.  The subclass may decide to expose some
        of the wrapped widgets by behaving like a ContainerWidget or
        WidgetDecoration, or it may hide them from outside access.
        """
        super().__init__()
        if not isinstance(w, Widget):
            obj_class_path = f'{w.__class__.__module__}.{w.__class__.__name__}'
            warnings.warn(f'{obj_class_path} is not subclass of Widget', DeprecationWarning, stacklevel=2)
        self._wrapped_widget = w

    @property
    def _w(self) -> WrappedWidget:
        return self._wrapped_widget

    @_w.setter
    def _w(self, new_widget: WrappedWidget) -> None:
        """
        Change the wrapped widget.  This is meant to be called
        only by subclasses.

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
        self._wrapped_widget = new_widget
        self._invalidate()

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