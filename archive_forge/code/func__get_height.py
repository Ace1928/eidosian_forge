from os.path import join, exists
from os import getcwd
from collections import defaultdict
from kivy.core import core_select_lib
from kivy.clock import Clock
from kivy.config import Config
from kivy.logger import Logger
from kivy.base import EventLoop, stopTouchApp
from kivy.modules import Modules
from kivy.event import EventDispatcher
from kivy.properties import ListProperty, ObjectProperty, AliasProperty, \
from kivy.utils import platform, reify, deprecated, pi_version
from kivy.context import get_current_context
from kivy.uix.behaviors import FocusBehavior
from kivy.setupconfig import USE_SDL2
from kivy.graphics.transformation import Matrix
from kivy.graphics.cgl import cgl_get_backend_name
def _get_height(self):
    """Rotated window height"""
    r = self._rotation
    _size = self._size
    if platform == 'win' or self._density != 1:
        _size = self._win._get_gl_size()
    kb = self.keyboard_height if self.softinput_mode == 'resize' else 0
    if r == 0 or r == 180:
        return _size[1] - kb
    return _size[0] - kb