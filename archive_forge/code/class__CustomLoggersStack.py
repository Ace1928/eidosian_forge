from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
class _CustomLoggersStack(object):

    def __init__(self):
        self._lock = threading.Lock()
        self._owning_thread_id = None
        self._stack = []

    @staticmethod
    def _get_stream_like_object(obj):
        if hasattr(obj, 'write'):
            return obj
        if hasattr(obj, '__call__'):
            return _StreamLikeWrapper(obj)
        raise CatBoostError('Expected callable object or stream-like object')

    def push(self, log_cout=None, log_cerr=None):
        with self._lock:
            if not self._stack:
                self._owning_thread_id = threading.current_thread().ident
            elif self._owning_thread_id != threading.current_thread().ident:
                if log_cout is not None or log_cerr is not None:
                    raise CatBoostError('CatBoost custom loggers have been already set in another thread. ' + ' Setting custom loggers from different threads is not currently supported')
                return

            def init_log(log, default, index_in_stack):
                if log is None:
                    if not self._stack:
                        return default
                    else:
                        return self._stack[-1][index_in_stack]
                return _CustomLoggersStack._get_stream_like_object(log)
            cout = init_log(log_cout, sys.stdout, 0)
            cerr = init_log(log_cout, sys.stderr, 1)
            _reset_logger()
            _set_logger(cout, cerr)
            self._stack.append((cout, cerr))

    def pop(self):
        with self._lock:
            if self._owning_thread_id != threading.current_thread().ident:
                return
            if not self._stack:
                raise RuntimeError('Attempt to pop from an empty stack')
            _reset_logger()
            if len(self._stack) != 1:
                _set_logger(*self._stack[-2])
            else:
                self._owning_thread_id = None
            self._stack.pop()