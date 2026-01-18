import builtins
import contextlib
import functools
import logging
import re
import sys
import warnings
from collections import namedtuple
from inspect import isclass, iscoroutinefunction, isgeneratorfunction
from multiprocessing import current_process, get_context
from multiprocessing.context import BaseContext
from os.path import basename, splitext
from threading import current_thread
from . import _asyncio_loop, _colorama, _defaults, _filters
from ._better_exceptions import ExceptionFormatter
from ._colorizer import Colorizer
from ._contextvars import ContextVar
from ._datetime import aware_now
from ._error_interceptor import ErrorInterceptor
from ._file_sink import FileSink
from ._get_frame import get_frame
from ._handler import Handler
from ._locks_machinery import create_logger_lock
from ._recattrs import RecordException, RecordFile, RecordLevel, RecordProcess, RecordThread
from ._simple_sinks import AsyncSink, CallableSink, StandardSink, StreamSink
def _change_activation(self, name, status):
    if not (name is None or isinstance(name, str)):
        raise TypeError("Invalid name, it should be a string (or None), not: '%s'" % type(name).__name__)
    with self._core.lock:
        enabled = self._core.enabled.copy()
        if name is None:
            for n in enabled:
                if n is None:
                    enabled[n] = status
            self._core.activation_none = status
            self._core.enabled = enabled
            return
        if name != '':
            name += '.'
        activation_list = [(n, s) for n, s in self._core.activation_list if n[:len(name)] != name]
        parent_status = next((s for n, s in activation_list if name[:len(n)] == n), None)
        if parent_status != status and (not (name == '' and status is True)):
            activation_list.append((name, status))

            def modules_depth(x):
                return x[0].count('.')
            activation_list.sort(key=modules_depth, reverse=True)
        for n in enabled:
            if n is not None and (n + '.')[:len(name)] == name:
                enabled[n] = status
        self._core.activation_list = activation_list
        self._core.enabled = enabled