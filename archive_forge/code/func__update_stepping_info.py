from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
import weakref
def _update_stepping_info(info):
    global _infos_stepping
    global _all_infos
    with _update_infos_lock:
        new_all_infos = set()
        for info in _all_infos:
            if info._get_related_thread() is not None:
                new_all_infos.add(info)
        _all_infos = new_all_infos
        new_stepping = set()
        for info in _all_infos:
            if info._is_stepping():
                new_stepping.add(info)
        _infos_stepping = new_stepping
    py_db = get_global_debugger()
    if py_db is not None and (not py_db.pydb_disposed):
        thread = info.weak_thread()
        if thread is not None:
            thread_id = get_thread_id(thread)
            _queue, event = py_db.get_internal_queue_and_event(thread_id)
            event.set()