from __future__ import annotations
import typing
from urwid.canvas import CompositeCanvas
from urwid.command_map import Command
from urwid.signals import connect_signal
from urwid.text_layout import calc_coords
from urwid.util import is_mouse_press
from .columns import Columns
from .constants import Align, WrapMode
from .text import Text
from .widget import WidgetError, WidgetWrap
class CheckBoxError(WidgetError):
    pass