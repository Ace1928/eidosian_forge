import os
import re
import sys
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_constants import get_global_debugger, IS_WINDOWS, IS_JYTHON, get_current_thread_id, \
from _pydev_bundle import pydev_log
from contextlib import contextmanager
from _pydevd_bundle import pydevd_constants, pydevd_defaults
from _pydevd_bundle.pydevd_defaults import PydevdCustomization
import ast
class _NewThreadStartupWithTrace:

    def __init__(self, original_func, args, kwargs):
        self.original_func = original_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        py_db = get_global_debugger()
        thread_id = None
        if py_db is not None:
            t = getattr(self.original_func, '__self__', getattr(self.original_func, 'im_self', None))
            if not isinstance(t, threading.Thread):
                t = threading.current_thread()
            if not getattr(t, 'is_pydev_daemon_thread', False):
                thread_id = get_current_thread_id(t)
                py_db.notify_thread_created(thread_id, t)
                _on_set_trace_for_new_thread(py_db)
            if getattr(py_db, 'thread_analyser', None) is not None:
                try:
                    from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_concurrency_logger import log_new_thread
                    log_new_thread(py_db, t)
                except:
                    sys.stderr.write('Failed to detect new thread for visualization')
        try:
            ret = self.original_func(*self.args, **self.kwargs)
        finally:
            if thread_id is not None:
                if py_db is not None:
                    py_db.disable_tracing()
                    py_db.notify_thread_not_alive(thread_id)
        return ret