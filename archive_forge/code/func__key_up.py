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
def _key_up(self, key, repeat=False):
    displayed_str, internal_str, internal_action, scale = key
    if internal_action in ('shift', 'shift_L', 'shift_R'):
        if self._selection:
            self._update_selection(True)
    elif internal_action == 'ctrl_L':
        self._ctrl_l = False
    elif internal_action == 'ctrl_R':
        self._ctrl_r = False
    elif internal_action == 'alt_L':
        self._alt_l = False
    elif internal_action == 'alt_R':
        self._alt_r = False