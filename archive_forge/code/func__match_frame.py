from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def _match_frame(self, filename, lineno):
    return self._match_frame_impl(filename, lineno) ^ (not self.inclusive)