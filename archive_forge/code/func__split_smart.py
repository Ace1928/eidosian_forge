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
def _split_smart(self, text):
    """
        Do a "smart" split. If not multiline, or if wrap is set,
        we are not doing smart split, just a split on line break.
        Otherwise, we are trying to split as soon as possible, to prevent
        overflow on the widget.
        """
    if not self.multiline or not self.do_wrap:
        lines = text.split(u'\n')
        lines_flags = [0] + [FL_IS_LINEBREAK] * (len(lines) - 1)
        return (lines, lines_flags)
    x = flags = 0
    line = []
    lines = []
    lines_flags = []
    _join = u''.join
    lines_append, lines_flags_append = (lines.append, lines_flags.append)
    padding_left = self.padding[0]
    padding_right = self.padding[2]
    width = self.width - padding_left - padding_right
    text_width = self._get_text_width
    _tab_width, _label_cached = (self.tab_width, self._label_cached)
    words_widths = {}
    for word in self._tokenize(text):
        is_newline = word == u'\n'
        try:
            w = words_widths[word]
        except KeyError:
            w = text_width(word, _tab_width, _label_cached)
            words_widths[word] = w
        if x + w > width and line or is_newline:
            lines_append(_join(line))
            lines_flags_append(flags)
            flags = 0
            line = []
            x = 0
        if is_newline:
            flags |= FL_IS_LINEBREAK
        elif width >= 1 and w > width:
            while w > width:
                split_width = split_pos = 0
                for c in word:
                    try:
                        cw = words_widths[c]
                    except KeyError:
                        cw = text_width(c, _tab_width, _label_cached)
                        words_widths[c] = cw
                    if split_width + cw > width:
                        break
                    split_width += cw
                    split_pos += 1
                if split_width == split_pos == 0:
                    break
                lines_append(word[:split_pos])
                lines_flags_append(flags)
                flags = FL_IS_WORDBREAK
                word = word[split_pos:]
                w -= split_width
            x = w
            line.append(word)
        else:
            x += w
            line.append(word)
    if line or flags & FL_IS_LINEBREAK:
        lines_append(_join(line))
        lines_flags_append(flags)
    return (lines, lines_flags)