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
def _get_effective_size(self):
    """On density=1 and non-ios / non-Windows displays,
        return :attr:`system_size`, else return scaled / rotated :attr:`size`.

        Used by MouseMotionEvent.update_graphics() and WindowBase.on_motion().
        """
    w, h = self.system_size
    if platform in ('ios', 'win') or self._density != 1:
        w, h = self.size
    return (w, h)