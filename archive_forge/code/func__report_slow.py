from __future__ import nested_scopes
import traceback
import warnings
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import _pydev_saved_modules
import signal
import os
import ctypes
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from urllib.parse import quote  # @UnresolvedImport
import time
import inspect
import sys
from _pydevd_bundle.pydevd_constants import USE_CUSTOM_SYS_CURRENT_FRAMES, IS_PYPY, SUPPORT_GEVENT, \
def _report_slow(self, compute_msg, *args):
    old = self._curr_time
    new = self._curr_time = time.time()
    diff = new - old
    if diff >= self.min_diff:
        py_db = get_global_debugger()
        if py_db is not None:
            msg = compute_msg(diff, *args)
            py_db.writer.add_command(py_db.cmd_factory.make_warning_message(msg))