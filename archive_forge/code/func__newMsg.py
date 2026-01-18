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
def _newMsg(self, msg, *args):
    msg = '  ' * (self._depth - 1) + msg
    if self._delayed:
        self._msgs.append((msg, args))
    else:
        self.flush()
        print(msg % args)