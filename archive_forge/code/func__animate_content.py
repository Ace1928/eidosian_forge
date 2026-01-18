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
def _animate_content(self):
    """Animate content to IME height.
        """
    kargs = self.keyboard_anim_args
    global Animation
    if not Animation:
        from kivy.animation import Animation
    if WindowBase._kanimation:
        WindowBase._kanimation.cancel(self)
    WindowBase._kanimation = kanim = Animation(_kheight=self.keyboard_height + self.keyboard_padding, d=kargs['d'], t=kargs['t'])
    kanim.bind(on_complete=self._free_kanimation)
    kanim.start(self)