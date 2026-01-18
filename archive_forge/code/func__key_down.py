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
def _key_down(self, key, repeat=False):
    displayed_str, internal_str, internal_action, scale = key
    if self._selection and internal_action in (None, 'del', 'backspace', 'enter') and (internal_action != 'enter' or self.multiline):
        self.delete_selection()
    elif internal_action == 'del':
        cursor = self.cursor
        self.do_cursor_movement('cursor_right')
        if cursor != self.cursor:
            self.do_backspace(mode='del')
    elif internal_action == 'backspace':
        self.do_backspace()
    if internal_action is None:
        self.insert_text(displayed_str)
    elif internal_action in ('shift', 'shift_L', 'shift_R'):
        if not self._selection:
            self._selection_from = self._selection_to = self.cursor_index()
            self._selection = True
        self._selection_finished = False
    elif internal_action == 'ctrl_L':
        self._ctrl_l = True
    elif internal_action == 'ctrl_R':
        self._ctrl_r = True
    elif internal_action == 'alt_L':
        self._alt_l = True
    elif internal_action == 'alt_R':
        self._alt_r = True
    elif internal_action.startswith('cursor_'):
        cc, cr = self.cursor
        self.do_cursor_movement(internal_action, self._ctrl_l or self._ctrl_r, self._alt_l or self._alt_r)
        if self._selection and (not self._selection_finished):
            self._selection_to = self.cursor_index()
            self._update_selection()
        else:
            self.cancel_selection()
    elif internal_action == 'enter':
        if self.multiline:
            self.insert_text(u'\n')
        else:
            self.dispatch('on_text_validate')
            if self.text_validate_unfocus:
                self.focus = False
    elif internal_action == 'escape':
        self.focus = False