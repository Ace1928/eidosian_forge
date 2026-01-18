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
def _get_cursor_pos(self):
    dy = self.line_height + self.line_spacing
    padding_left = self.padding[0]
    padding_top = self.padding[1]
    padding_right = self.padding[2]
    left = self.x + padding_left
    top = self.top - padding_top
    y = top + self.scroll_y
    y -= self.cursor_row * dy
    halign = self.halign
    viewport_width = self.width - padding_left - padding_right
    cursor_offset = self.cursor_offset()
    base_dir = self.base_direction or self._resolved_base_dir
    auto_halign_r = halign == 'auto' and base_dir and ('rtl' in base_dir)
    if halign == 'center':
        row_width = self._get_row_width(self.cursor_row)
        x = left + max(0, (viewport_width - row_width) // 2) + cursor_offset - self.scroll_x
    elif halign == 'right' or auto_halign_r:
        row_width = self._get_row_width(self.cursor_row)
        x = left + max(0, viewport_width - row_width) + cursor_offset - self.scroll_x
    else:
        x = left + cursor_offset - self.scroll_x
    return (x, y)