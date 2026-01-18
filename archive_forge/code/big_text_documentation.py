from __future__ import annotations
import typing
from urwid.canvas import CanvasJoin, CompositeCanvas, TextCanvas
from urwid.util import decompose_tagmarkup
from .constants import Sizing
from .widget import Widget, fixed_size

        Returns (text, attributes).
        