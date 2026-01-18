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
def _run_with_interrupt_thread(original_func, py_db, curr_thread, frame, expression, is_exec):
    on_interrupt_threads = None
    timeout_tracker = py_db.timeout_tracker
    interrupt_thread_timeout = pydevd_constants.PYDEVD_INTERRUPT_THREAD_TIMEOUT
    if interrupt_thread_timeout > 0:
        on_interrupt_threads = pydevd_timeout.create_interrupt_this_thread_callback()
        pydev_log.info('Doing evaluate with interrupt threads timeout: %s.', interrupt_thread_timeout)
    if on_interrupt_threads is None:
        return original_func(py_db, frame, expression, is_exec)
    else:
        with timeout_tracker.call_on_timeout(interrupt_thread_timeout, on_interrupt_threads):
            return original_func(py_db, frame, expression, is_exec)