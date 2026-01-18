import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
def _thread_profile(self, f, *args, **kwds):
    thr = _thread.get_ident()
    self._g_threadmap[thr] = p = Profiler()
    p.enable(subcalls=True, builtins=True)