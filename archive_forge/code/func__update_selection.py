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
def _update_selection(self, finished=False):
    """Update selection text and order of from/to if finished is True.
        Can be called multiple times until finished is True.
        """
    a, b = (int(self._selection_from), int(self._selection_to))
    if a > b:
        a, b = (b, a)
    self._selection_finished = finished
    _selection_text = self.text[a:b]
    self.selection_text = '' if not self.allow_copy else self.password_mask * (b - a) if self.password else _selection_text
    if not finished:
        self._selection = True
    else:
        self._selection = bool(len(_selection_text))
        self._selection_touch = None
    if a == 0:
        self._update_graphics_selection()