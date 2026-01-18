import contextlib
import logging
import threading
import time
from tensorboard.util import tb_logging
class _ThreadLocalStore(threading.local):

    def __init__(self):
        self.nesting_level = 0