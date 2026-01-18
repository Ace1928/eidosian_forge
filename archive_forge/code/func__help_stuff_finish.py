import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
@staticmethod
def _help_stuff_finish(inqueue, task_handler, size):
    try:
        while True:
            inqueue.get(block=False)
    except queue.Empty:
        pass
    for i in range(size):
        inqueue.put(None)