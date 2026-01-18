from os.path import join
import sys
from typing import Optional
from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.base import EventLoop
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import WindowBase
from kivy.input.provider import MotionEventProvider
from kivy.input.motionevent import MotionEvent
from kivy.resources import resource_find
from kivy.utils import platform, deprecated
from kivy.compat import unichr
from collections import deque
def _collide_and_dispatch_cursor_enter(self, x, y):
    w, h = self._win.window_size
    if 0 <= x < w and 0 <= y < h:
        self._mouse_x, self._mouse_y = self._fix_mouse_pos(x, y)
        if not self._cursor_entered:
            self._cursor_entered = True
            self.dispatch('on_cursor_enter')
        return True