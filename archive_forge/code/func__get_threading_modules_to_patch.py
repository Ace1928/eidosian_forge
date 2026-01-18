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
def _get_threading_modules_to_patch():
    threading_modules_to_patch = []
    try:
        import thread as _thread
    except:
        import _thread
    threading_modules_to_patch.append(_thread)
    threading_modules_to_patch.append(threading)
    return threading_modules_to_patch