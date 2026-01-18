from kivy import kivy_data_dir
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.core.image import ImageLoader, Image
from kivy.config import Config
from kivy.utils import platform
from collections import deque
from time import sleep
from os.path import join
from os import write, close, unlink, environ
import threading
import mimetypes
def _wait_for_resume(self):
    while self._running and self._paused:
        self._resume_cond.acquire()
        self._resume_cond.wait(0.25)
        self._resume_cond.release()