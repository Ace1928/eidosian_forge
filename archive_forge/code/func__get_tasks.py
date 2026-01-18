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
def _get_tasks(func, it, size):
    it = iter(it)
    while 1:
        x = tuple(itertools.islice(it, size))
        if not x:
            return
        yield (func, x)