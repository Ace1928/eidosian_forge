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
def _evaluate_with_timeouts(original_func):
    """
    Provides a decorator that wraps the original evaluate to deal with slow evaluates.

    If some evaluation is too slow, we may show a message, resume threads or interrupt them
    as needed (based on the related configurations).
    """

    @functools.wraps(original_func)
    def new_func(py_db, frame, expression, is_exec):
        if py_db is None:
            pydev_log.critical('_evaluate_with_timeouts called without py_db!')
            return original_func(py_db, frame, expression, is_exec)
        warn_evaluation_timeout = pydevd_constants.PYDEVD_WARN_EVALUATION_TIMEOUT
        curr_thread = threading.current_thread()

        def on_warn_evaluation_timeout():
            py_db.writer.add_command(py_db.cmd_factory.make_evaluation_timeout_msg(py_db, expression, curr_thread))
        timeout_tracker = py_db.timeout_tracker
        with timeout_tracker.call_on_timeout(warn_evaluation_timeout, on_warn_evaluation_timeout):
            return _run_with_unblock_threads(original_func, py_db, curr_thread, frame, expression, is_exec)
    return new_func