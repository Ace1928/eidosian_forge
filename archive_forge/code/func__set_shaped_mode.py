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
def _set_shaped_mode(self, value):
    self._set_shape(shape_image=self.shape_image, mode=value, cutoff=self.shape_cutoff, color_key=self.shape_color_key)
    return self._win.get_shaped_mode()