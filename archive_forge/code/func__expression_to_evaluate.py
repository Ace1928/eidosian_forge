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
@lru_cache(3)
def _expression_to_evaluate(expression):
    keepends = True
    lines = expression.splitlines(keepends)
    chars_to_strip = 0
    for line in lines:
        if line.strip():
            for c in iter_chars(line):
                if c.isspace():
                    chars_to_strip += 1
                else:
                    break
            break
    if chars_to_strip:
        proceed = True
        new_lines = []
        for line in lines:
            if not proceed:
                break
            for c in iter_chars(line[:chars_to_strip]):
                if not c.isspace():
                    proceed = False
                    break
            new_lines.append(line[chars_to_strip:])
        if proceed:
            if isinstance(expression, bytes):
                expression = b''.join(new_lines)
            else:
                expression = u''.join(new_lines)
    return expression