from asyncio import iscoroutinefunction
from contextlib import contextmanager
from functools import partial, wraps
from types import coroutine
import builtins
import inspect
import linecache
import logging
import os
import io
import pdb
import subprocess
import sys
import time
import traceback
import warnings
import psutil
class LineProfiler(object):
    """ A profiler that records the amount of memory for each line """

    def __init__(self, **kw):
        include_children = kw.get('include_children', False)
        backend = kw.get('backend', 'psutil')
        self.code_map = CodeMap(include_children=include_children, backend=backend)
        self.enable_count = 0
        self.max_mem = kw.get('max_mem', None)
        self.prevlines = []
        self.backend = choose_backend(kw.get('backend', None))
        self.prev_lineno = None

    def __call__(self, func=None, precision=1):
        if func is not None:
            self.add_function(func)
            f = self.wrap_function(func)
            f.__module__ = func.__module__
            f.__name__ = func.__name__
            f.__doc__ = func.__doc__
            f.__dict__.update(getattr(func, '__dict__', {}))
            return f
        else:

            def inner_partial(f):
                return self.__call__(f, precision=precision)
            return inner_partial

    def add_function(self, func):
        """ Record line profiling information for the given Python function.
        """
        try:
            code = func.__code__
        except AttributeError:
            warnings.warn('Could not extract a code object for the object %r' % func)
        else:
            self.code_map.add(code)

    @contextmanager
    def _count_ctxmgr(self):
        self.enable_by_count()
        try:
            yield
        finally:
            self.disable_by_count()

    def wrap_function(self, func):
        """ Wrap a function to profile it.
        """
        if iscoroutinefunction(func):

            @coroutine
            def f(*args, **kwargs):
                with self._count_ctxmgr():
                    res = (yield from func(*args, **kwargs))
                    return res
        else:

            def f(*args, **kwds):
                with self._count_ctxmgr():
                    return func(*args, **kwds)
        return f

    def runctx(self, cmd, globals, locals):
        """ Profile a single executable statement in the given namespaces.
        """
        self.enable_by_count()
        try:
            exec(cmd, globals, locals)
        finally:
            self.disable_by_count()
        return self

    def enable_by_count(self):
        """ Enable the profiler if it hasn't been enabled before.
        """
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """ Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def trace_memory_usage(self, frame, event, arg):
        """Callback for sys.settrace"""
        if frame.f_code in self.code_map:
            if event == 'call':
                self.prevlines.append(frame.f_lineno)
            elif event == 'line':
                self.code_map.trace(frame.f_code, self.prevlines[-1], self.prev_lineno)
                self.prev_lineno = self.prevlines[-1]
                self.prevlines[-1] = frame.f_lineno
            elif event == 'return':
                lineno = self.prevlines.pop()
                self.code_map.trace(frame.f_code, lineno, self.prev_lineno)
                self.prev_lineno = lineno
        if self._original_trace_function is not None:
            self._original_trace_function(frame, event, arg)
        return self.trace_memory_usage

    def trace_max_mem(self, frame, event, arg):
        if event in ('line', 'return') and frame.f_code in self.code_map:
            c = _get_memory(-1, self.backend, filename=frame.f_code.co_filename)
            if c >= self.max_mem:
                t = 'Current memory {0:.2f} MiB exceeded the maximum of {1:.2f} MiB\n'.format(c, self.max_mem)
                sys.stdout.write(t)
                sys.stdout.write('Stepping into the debugger \n')
                frame.f_lineno -= 2
                p = pdb.Pdb()
                p.quitting = False
                p.stopframe = frame
                p.returnframe = None
                p.stoplineno = frame.f_lineno - 3
                p.botframe = None
                return p.trace_dispatch
        if self._original_trace_function is not None:
            self._original_trace_function(frame, event, arg)
        return self.trace_max_mem

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        self._original_trace_function = sys.gettrace()
        if self.max_mem is not None:
            sys.settrace(self.trace_max_mem)
        else:
            sys.settrace(self.trace_memory_usage)

    def disable(self):
        sys.settrace(self._original_trace_function)