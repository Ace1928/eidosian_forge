from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def _pydevd_log_exception(msg='', *args):
    if msg or args:
        _pydevd_log(0, msg, *args)
    try:
        initialize_debug_stream()
        traceback.print_exc(file=_LoggingGlobals._debug_stream)
        _LoggingGlobals._debug_stream.flush()
    except:
        raise