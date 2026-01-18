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
class BoxWidget(Widget):
    """
    Deprecated.  Inherit from Widget and add:

        _sizing = frozenset(['box'])
        _selectable = True

    at the top of your class definition instead.

    Base class of width and height constrained widgets such as
    the top level widget attached to the display object
    """
    _selectable = True
    _sizing = frozenset([Sizing.BOX])

    def __init__(self, *args, **kwargs):
        warnings.warn("\n            BoxWidget is deprecated. Inherit from Widget and add:\n\n                _sizing = frozenset(['box'])\n                _selectable = True\n\n            at the top of your class definition instead.", DeprecationWarning, stacklevel=3)
        super().__init__()

    def render(self, size: tuple[int, int], focus: bool=False) -> Canvas:
        """
        All widgets must implement this function.
        """
        raise NotImplementedError()