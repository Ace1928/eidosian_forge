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
def _update_graphics_selection(self):
    if not self._selection:
        return
    padding_left, padding_top, padding_right, padding_bottom = self.padding
    rects = self._lines_rects
    label_cached = self._label_cached
    lines = self._lines
    tab_width = self.tab_width
    top = self.top
    get_text_width = self._get_text_width
    get_cursor_from_index = self.get_cursor_from_index
    draw_selection = self._draw_selection
    canvas_add = self.canvas.add
    selection_color = self.selection_color
    a, b = sorted((self._selection_from, self._selection_to))
    selection_start_col, selection_start_row = get_cursor_from_index(a)
    selection_end_col, selection_end_row = get_cursor_from_index(b)
    dy = self.line_height + self.line_spacing
    x = self.x
    y = top - padding_top + self.scroll_y - selection_start_row * dy
    width = self.width
    miny = self.y + padding_bottom
    maxy = top - padding_top + dy
    self.canvas.remove_group('selection')
    first_visible_line = math.floor(self.scroll_y / dy)
    last_visible_line = math.ceil((self.scroll_y + maxy - miny) / dy)
    width_minus_padding = width - (padding_right + padding_left)
    for line_num, rect in enumerate(islice(rects, max(selection_start_row, first_visible_line), min(selection_end_row + 1, last_visible_line - 1)), start=max(selection_start_row, first_visible_line)):
        draw_selection(rect.pos, rect.size, line_num, (selection_start_col, selection_start_row), (selection_end_col, selection_end_row), lines, get_text_width, tab_width, label_cached, width_minus_padding, padding_left, padding_right, x, canvas_add, selection_color)
    self._position_handles('both')