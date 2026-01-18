from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def _filter_trace(self, include_filters, exclude_filters, trace):
    if include_filters:
        if not any((trace_filter._match(trace) for trace_filter in include_filters)):
            return False
    if exclude_filters:
        if any((not trace_filter._match(trace) for trace_filter in exclude_filters)):
            return False
    return True