from threading import Thread
from queue import Queue, Empty, Full
from kivy.clock import Clock, mainthread
from kivy.logger import Logger
from kivy.core.video import VideoBase
from kivy.graphics import Rectangle, BindTexture
from kivy.graphics.texture import Texture
from kivy.graphics.fbo import Fbo
from kivy.weakmethod import WeakMethod
import time
@mainthread
def _finish_setup(self):
    if self._ffplayer is not None:
        self._ffplayer.set_volume(self._volume)
        self._ffplayer.set_pause(self._state == 'paused')
        self._wakeup_thread()