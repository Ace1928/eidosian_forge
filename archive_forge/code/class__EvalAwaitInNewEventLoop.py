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
class _EvalAwaitInNewEventLoop(PyDBDaemonThread):

    def __init__(self, py_db, compiled, updated_globals, updated_locals):
        PyDBDaemonThread.__init__(self, py_db)
        self._compiled = compiled
        self._updated_globals = updated_globals
        self._updated_locals = updated_locals
        self.evaluated_value = None
        self.exc = None

    async def _async_func(self):
        return await eval(self._compiled, self._updated_locals, self._updated_globals)

    def _on_run(self):
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.evaluated_value = asyncio.run(self._async_func())
        except:
            self.exc = sys.exc_info()