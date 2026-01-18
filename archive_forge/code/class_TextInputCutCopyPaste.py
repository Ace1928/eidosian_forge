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
class TextInputCutCopyPaste(Bubble):
    textinput = ObjectProperty(None)
    ' Holds a reference to the TextInput this Bubble belongs to.\n    '
    but_cut = ObjectProperty(None)
    but_copy = ObjectProperty(None)
    but_paste = ObjectProperty(None)
    but_selectall = ObjectProperty(None)
    matrix = ObjectProperty(None)
    _check_parent_ev = None

    def __init__(self, **kwargs):
        self.mode = 'normal'
        super().__init__(**kwargs)
        self._check_parent_ev = Clock.schedule_interval(self._check_parent, 0.5)
        self.matrix = self.textinput.get_window_matrix()
        with self.canvas.before:
            Callback(self.update_transform)
            PushMatrix()
            self.transform = Transform()
        with self.canvas.after:
            PopMatrix()

    def update_transform(self, cb):
        m = self.textinput.get_window_matrix()
        if self.matrix != m:
            self.matrix = m
            self.transform.identity()
            self.transform.transform(self.matrix)

    def transform_touch(self, touch):
        matrix = self.matrix.inverse()
        touch.apply_transform_2d(lambda x, y: matrix.transform_point(x, y, 0)[:2])

    def on_touch_down(self, touch):
        try:
            touch.push()
            self.transform_touch(touch)
            if self.collide_point(*touch.pos):
                FocusBehavior.ignored_touch.append(touch)
            return super().on_touch_down(touch)
        finally:
            touch.pop()

    def on_touch_up(self, touch):
        try:
            touch.push()
            self.transform_touch(touch)
            for child in self.content.children:
                if ref(child) in touch.grab_list:
                    touch.grab_current = child
                    break
            return super().on_touch_up(touch)
        finally:
            touch.pop()

    def on_textinput(self, instance, value):
        global Clipboard
        if value and (not Clipboard) and (not _is_desktop):
            value._ensure_clipboard()

    def _check_parent(self, dt):
        parent = self.textinput
        while parent is not None:
            if parent == parent.parent:
                break
            parent = parent.parent
        if parent is None:
            self._check_parent_ev.cancel()
            if self.textinput:
                self.textinput._hide_cut_copy_paste()

    def on_parent(self, instance, value):
        parent = self.textinput
        mode = self.mode
        if parent:
            self.content.clear_widgets()
            if mode == 'paste':
                self.but_selectall.opacity = 1
                widget_list = [self.but_selectall]
                if not parent.readonly:
                    widget_list.append(self.but_paste)
            elif parent.readonly:
                widget_list = (self.but_copy,)
            else:
                widget_list = (self.but_cut, self.but_copy, self.but_paste)
            for widget in widget_list:
                self.content.add_widget(widget)

    def do(self, action):
        textinput = self.textinput
        if action == 'cut':
            textinput._cut(textinput.selection_text)
        elif action == 'copy':
            textinput.copy()
        elif action == 'paste':
            textinput.paste()
        elif action == 'selectall':
            textinput.select_all()
            self.mode = ''
            anim = Animation(opacity=0, d=0.333)
            anim.bind(on_complete=lambda *args: self.on_parent(self, self.parent))
            anim.start(self.but_selectall)
            return
        self.hide()

    def hide(self):
        parent = self.parent
        if not parent:
            return
        anim = Animation(opacity=0, d=0.225)
        anim.bind(on_complete=lambda *args: parent.remove_widget(self))
        anim.start(self)