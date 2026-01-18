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
def _position_handles(self, *args, **kwargs):
    if not self.text:
        return
    mode = kwargs.get('mode', 'both')
    lh = self.line_height
    handle_middle = self._handle_middle
    if handle_middle:
        hp_mid = self.cursor_pos
        pos = self.to_local(*hp_mid, relative=True)
        handle_middle.x = pos[0] - handle_middle.width / 2
        handle_middle.top = max(self.padding[3], min(self.height - self.padding[1], pos[1] - lh))
    if mode[0] == 'm':
        return
    group = self.canvas.get_group('selection')
    if not group:
        return
    EventLoop.window.remove_widget(self._handle_middle)
    handle_left = self._handle_left
    if not handle_left:
        return
    hp_left = group[2].pos
    handle_left.pos = self.to_local(*hp_left, relative=True)
    handle_left.x -= handle_left.width
    handle_left.y -= handle_left.height
    handle_right = self._handle_right
    last_rect = group[-1]
    hp_right = (last_rect.pos[0], last_rect.pos[1])
    x, y = self.to_local(*hp_right, relative=True)
    handle_right.x = x + last_rect.size[0]
    handle_right.y = y - handle_right.height