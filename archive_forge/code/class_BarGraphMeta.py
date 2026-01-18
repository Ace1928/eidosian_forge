from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
class BarGraphMeta(WidgetMeta):
    """
    Detect subclass get_data() method and dynamic change to
    get_data() method and disable caching in these cases.

    This is for backwards compatibility only, new programs
    should use set_data() instead of overriding get_data().
    """

    def __init__(cls, name, bases, d):
        super().__init__(name, bases, d)
        if 'get_data' in d:
            cls.render = nocache_widget_render(cls)
            cls._get_data = cls.get_data
        cls.get_data = property(lambda self: self._get_data, nocache_bargraph_get_data)