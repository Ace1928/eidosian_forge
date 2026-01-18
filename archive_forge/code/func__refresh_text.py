import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def _refresh_text(self, text, *largs):
    """
        Refresh all the lines from a new text.
        By using cache in internal functions, this method should be fast.
        """
    mode = 'all'
    if len(largs) > 1:
        mode, start, finish, _lines, _lines_flags, len_lines = largs
        cursor = None
    else:
        cursor = self.cursor_index()
        _lines, self._lines_flags = self._split_smart(text)
    _lines_labels = []
    _line_rects = []
    _create_label = self._create_line_label
    for x in _lines:
        lbl = _create_label(x)
        _lines_labels.append(lbl)
        _line_rects.append(Rectangle(size=lbl.size))
    if mode == 'all':
        self._lines_labels = _lines_labels
        self._lines_rects = _line_rects
        self._lines[:] = _lines
    elif mode == 'del':
        if finish > start:
            self._insert_lines(start, finish + 1, len_lines, _lines_flags, _lines, _lines_labels, _line_rects)
    elif mode == 'insert':
        self._insert_lines(start, finish + 1, len_lines, _lines_flags, _lines, _lines_labels, _line_rects)
    min_line_ht = self._label_cached.get_extents('_')[1]
    self.line_height = max(_lines_labels[0].height, min_line_ht)
    row = self.cursor_row
    self.cursor = self.get_cursor_from_index(self.cursor_index() if cursor is None else cursor)
    if self.cursor_row != row:
        self.scroll_x = 0
    self._trigger_update_graphics()