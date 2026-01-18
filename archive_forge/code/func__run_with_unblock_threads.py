import pickle
from _pydevd_bundle.pydevd_constants import get_frame, get_current_thread_id, \
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate, get_type, var_to_xml
from _pydev_bundle import pydev_log
import functools
from _pydevd_bundle.pydevd_thread_lifecycle import resume_threads, mark_thread_suspended, suspend_all_threads
from _pydevd_bundle.pydevd_comm_constants import CMD_SET_BREAK
import sys  # @Reimport
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_save_locals, pydevd_timeout, pydevd_constants
from _pydev_bundle.pydev_imports import Exec, execfile
from _pydevd_bundle.pydevd_utils import to_string
import inspect
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_save_locals import update_globals_and_locals
from functools import lru_cache
def _run_with_unblock_threads(original_func, py_db, curr_thread, frame, expression, is_exec):
    on_timeout_unblock_threads = None
    timeout_tracker = py_db.timeout_tracker
    if py_db.multi_threads_single_notification:
        unblock_threads_timeout = pydevd_constants.PYDEVD_UNBLOCK_THREADS_TIMEOUT
    else:
        unblock_threads_timeout = -1
    if unblock_threads_timeout >= 0:
        pydev_log.info('Doing evaluate with unblock threads timeout: %s.', unblock_threads_timeout)
        tid = get_current_thread_id(curr_thread)

        def on_timeout_unblock_threads():
            on_timeout_unblock_threads.called = True
            pydev_log.info('Resuming threads after evaluate timeout.')
            resume_threads('*', except_thread=curr_thread)
            py_db.threads_suspended_single_notification.on_thread_resume(tid, curr_thread)
        on_timeout_unblock_threads.called = False
    try:
        if on_timeout_unblock_threads is None:
            return _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec)
        else:
            with timeout_tracker.call_on_timeout(unblock_threads_timeout, on_timeout_unblock_threads):
                return _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec)
    finally:
        if on_timeout_unblock_threads is not None and on_timeout_unblock_threads.called:
            mark_thread_suspended(curr_thread, CMD_SET_BREAK)
            py_db.threads_suspended_single_notification.increment_suspend_time()
            suspend_all_threads(py_db, except_thread=curr_thread)
            py_db.threads_suspended_single_notification.on_thread_suspend(tid, curr_thread, CMD_SET_BREAK)