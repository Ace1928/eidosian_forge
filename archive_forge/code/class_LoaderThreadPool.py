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
class LoaderThreadPool(LoaderBase):

    def __init__(self):
        super(LoaderThreadPool, self).__init__()
        self.pool = None

    def start(self):
        super(LoaderThreadPool, self).start()
        self.pool = _ThreadPool(self._num_workers)
        Clock.schedule_interval(self.run, 0)

    def stop(self):
        super(LoaderThreadPool, self).stop()
        Clock.unschedule(self.run)
        self.pool.stop()

    def run(self, *largs):
        while self._running:
            try:
                parameters = self._q_load.pop()
            except:
                return
            self.pool.add_task(self._load, parameters)