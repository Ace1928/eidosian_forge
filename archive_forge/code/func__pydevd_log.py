from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def _pydevd_log(level, msg, *args):
    """
    Levels are:

    0 most serious warnings/errors (always printed)
    1 warnings/significant events
    2 informational trace
    3 verbose mode
    """
    if level <= DebugInfoHolder.DEBUG_TRACE_LEVEL:
        try:
            try:
                if args:
                    msg = msg % args
            except:
                msg = '%s - %s' % (msg, args)
            if LOG_TIME:
                global _last_log_time
                new_log_time = time.time()
                time_diff = new_log_time - _last_log_time
                _last_log_time = new_log_time
                msg = '%.2fs - %s\n' % (time_diff, msg)
            else:
                msg = '%s\n' % (msg,)
            if _LOG_PID:
                msg = '<%s> - %s\n' % (os.getpid(), msg)
            try:
                try:
                    initialize_debug_stream()
                    _LoggingGlobals._debug_stream.write(msg)
                except TypeError:
                    if isinstance(msg, bytes):
                        msg = msg.decode('utf-8', 'replace')
                        _LoggingGlobals._debug_stream.write(msg)
            except UnicodeEncodeError:
                encoding = getattr(_LoggingGlobals._debug_stream, 'encoding', 'ascii')
                msg = msg.encode(encoding, 'backslashreplace')
                msg = msg.decode(encoding)
                _LoggingGlobals._debug_stream.write(msg)
            _LoggingGlobals._debug_stream.flush()
        except:
            pass
        return True