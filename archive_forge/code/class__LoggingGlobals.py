from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
class _LoggingGlobals(object):
    _warn_once_map = {}
    _debug_stream_filename = None
    _debug_stream = NULL
    _debug_stream_initialized = False
    _initialize_lock = ForkSafeLock()