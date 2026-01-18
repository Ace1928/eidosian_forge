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
class DisabledProfiler(object):

    def __init__(self, *args, **kwds):
        pass

    def __call__(self, *args):
        pass

    def finish(self):
        pass

    def mark(self, msg=None):
        pass