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
def _expand_range(self, ifrom, ito=None):
    if ito is None:
        ito = ifrom
    rfrom = self.get_cursor_from_index(ifrom)[1]
    rtcol, rto = self.get_cursor_from_index(ito)
    rfrom, rto = self._expand_rows(rfrom, rto + 1 if rtcol else rto)
    return (self.cursor_index((0, rfrom)), self.cursor_index((0, rto)))