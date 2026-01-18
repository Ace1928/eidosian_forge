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
def _set_minimum_size(self, *args):
    minimum_width = self.minimum_width
    minimum_height = self.minimum_height
    if minimum_width and minimum_height:
        self._win.set_minimum_size(minimum_width, minimum_height)
    elif minimum_width or minimum_height:
        Logger.warning('Both Window.minimum_width and Window.minimum_height must be bigger than 0 for the size restriction to take effect.')