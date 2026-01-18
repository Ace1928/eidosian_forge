from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
class ThreadColor(object):
    """
    Wrapper on stdout/stderr that colors text by the current thread ID.

    *stream* must be 'stdout' or 'stderr'.
    """
    colors = {}
    lock = Mutex()

    def __init__(self, stream):
        self.stream = getattr(sys, stream)
        self.err = stream == 'stderr'
        setattr(sys, stream, self)

    def write(self, msg):
        with self.lock:
            cprint.cprint(self.stream, self.color(), msg, -1, stderr=self.err)

    def flush(self):
        with self.lock:
            self.stream.flush()

    def color(self):
        tid = threading.current_thread()
        if tid not in self.colors:
            c = len(self.colors) % 15 + 1
            self.colors[tid] = c
        return self.colors[tid]