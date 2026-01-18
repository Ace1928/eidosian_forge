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
def _get_text_width(self, text, tab_width, _label_cached):
    """Return the width of a text, according to the current line options"""
    kw = self._get_line_options()
    try:
        cid = u'{}\x00{}\x00{}'.format(text, self.password, kw)
    except UnicodeDecodeError:
        cid = '{}\x00{}\x00{}'.format(text, self.password, kw)
    width = Cache_get('textinput.width', cid)
    if width:
        return width
    if not _label_cached:
        _label_cached = self._label_cached
    text = text.replace('\t', ' ' * tab_width)
    if not self.password:
        width = _label_cached.get_extents(text)[0]
    else:
        width = _label_cached.get_extents(self.password_mask * len(text))[0]
    Cache_append('textinput.width', cid, width)
    return width