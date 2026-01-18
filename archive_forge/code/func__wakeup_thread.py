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
def _wakeup_thread(self):
    try:
        self._wakeup_queue.put(None, False)
    except Full:
        pass