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
def _expand_rows(self, rfrom, rto=None):
    if rto is None or rto == rfrom:
        rto = rfrom + 1
    lines = self._lines
    flags = list(reversed(self._lines_flags))
    while rfrom > 0 and (not flags[rfrom - 1] & FL_IS_NEWLINE):
        rfrom -= 1
    rmax = len(lines) - 1
    while 0 < rto < rmax and (not flags[rto - 1] & FL_IS_NEWLINE):
        rto += 1
    return (max(0, rfrom), min(rmax, rto))