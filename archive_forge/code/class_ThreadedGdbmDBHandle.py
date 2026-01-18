import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
class ThreadedGdbmDBHandle(GdbmDBHandle):
    """Like GdbmDBHandle, but handles multi-threaded access."""

    def __init__(self, fn, mode, max_age=None, bound=None):
        self.db_lock = threading.Lock()
        GdbmDBHandle.__init__(self, fn, mode, max_age=max_age)

    def apply_method(self, method, varargs=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with self.db_lock:
            return GdbmDBHandle.apply_method(self, method, varargs=varargs, kwargs=kwargs)