from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
import weakref
def _get_related_thread(self):
    if self.pydev_notify_kill:
        return None
    if self.weak_thread is None:
        return None
    thread = self.weak_thread()
    if thread is None:
        return False
    if thread._is_stopped:
        return None
    if thread._ident is None:
        pydev_log.critical('thread._ident is None in _get_related_thread!')
        return None
    if threading._active.get(thread._ident) is not thread:
        return None
    return thread