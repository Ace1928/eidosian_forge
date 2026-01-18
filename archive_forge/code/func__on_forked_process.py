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
def _on_forked_process(setup_tracing=True):
    pydevd_constants.after_fork()
    pydev_log.initialize_debug_stream(reinitialize=True)
    if setup_tracing:
        pydev_log.debug('pydevd on forked process: %s', os.getpid())
    import pydevd
    pydevd.threadingCurrentThread().__pydevd_main_thread = True
    pydevd.settrace_forked(setup_tracing=setup_tracing)