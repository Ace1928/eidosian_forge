from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_constants import thread_get_ident, IS_CPYTHON, NULL
import ctypes
import time
from _pydev_bundle import pydev_log
import weakref
from _pydevd_bundle.pydevd_utils import is_current_thread_main_thread
from _pydevd_bundle import pydevd_utils
class _OnTimeoutHandle(object):

    def __init__(self, tracker, abs_timeout, on_timeout, kwargs):
        self._str = '_OnTimeoutHandle(%s)' % (on_timeout,)
        self._tracker = weakref.ref(tracker)
        self.abs_timeout = abs_timeout
        self.on_timeout = on_timeout
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.disposed = False

    def exec_on_timeout(self):
        kwargs = self.kwargs
        on_timeout = self.on_timeout
        if not self.disposed:
            self.disposed = True
            self.kwargs = None
            self.on_timeout = None
            try:
                if _DEBUG:
                    pydev_log.critical('pydevd_timeout: Calling on timeout: %s with kwargs: %s', on_timeout, kwargs)
                on_timeout(**kwargs)
            except Exception:
                pydev_log.exception('pydevd_timeout: Exception on callback timeout.')

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        tracker = self._tracker()
        if tracker is None:
            lock = NULL
        else:
            lock = tracker._lock
        with lock:
            self.disposed = True
            self.kwargs = None
            self.on_timeout = None

    def __str__(self):
        return self._str
    __repr__ = __str__