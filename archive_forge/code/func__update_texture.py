from kivy.graphics.texture import Texture
from kivy.core.video import VideoBase
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.compat import PY2
from threading import Lock
from functools import partial
from os.path import realpath
from weakref import ref
def _update_texture(self, buf):
    width, height, data = buf
    if not self._texture:
        self._texture = Texture.create(size=(width, height), colorfmt='rgb')
        self._texture.flip_vertical()
        self.dispatch('on_load')
    if self._texture:
        self._texture.blit_buffer(data, size=(width, height), colorfmt='rgb')