from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def _get_stack_str(frame):
    msg = '\nIf this is needed, please check: ' + '\nhttp://pydev.blogspot.com/2007/06/why-cant-pydev-debugger-work-with.html' + '\nto see how to restore the debug tracing back correctly.\n'
    if TracingFunctionHolder._traceback_limit:
        s = StringIO()
        s.write('Call Location:\n')
        traceback.print_stack(f=frame, limit=TracingFunctionHolder._traceback_limit, file=s)
        msg = msg + s.getvalue()
    return msg