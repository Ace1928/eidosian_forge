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
class _ThreadPool(object):
    """Pool of threads consuming tasks from a queue
        """

    def __init__(self, num_threads):
        super(_ThreadPool, self).__init__()
        self.running = True
        self.tasks = queue.Queue()
        for _ in range(num_threads):
            _Worker(self, self.tasks)

    def add_task(self, func, *args, **kargs):
        """Add a task to the queue
            """
        self.tasks.put((func, args, kargs))

    def stop(self):
        self.running = False
        self.tasks.join()