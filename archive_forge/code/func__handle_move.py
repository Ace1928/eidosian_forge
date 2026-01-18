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
def _handle_move(self, instance, touch):
    if touch.grab_current != instance:
        return
    get_cursor = self.get_cursor_from_xy
    handle_right = self._handle_right
    handle_left = self._handle_left
    handle_middle = self._handle_middle
    try:
        touch.push()
        touch.apply_transform_2d(self.to_widget)
        x, y = touch.pos
    finally:
        touch.pop()
    cursor = get_cursor(x, y + instance._touch_diff + self.line_height / 2)
    self.cursor = cursor
    if instance != touch.grab_current:
        return
    if instance == handle_middle:
        self._position_handles(mode='middle')
        return
    cindex = self.cursor_index()
    if instance == handle_left:
        self._selection_from = cindex
    elif instance == handle_right:
        self._selection_to = cindex
    self._update_selection()
    self._trigger_update_graphics()
    self._trigger_position_handles()