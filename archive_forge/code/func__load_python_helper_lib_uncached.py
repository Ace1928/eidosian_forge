from _pydevd_bundle.pydevd_constants import get_frame, IS_CPYTHON, IS_64BIT_PROCESS, IS_WINDOWS, \
from _pydev_bundle._pydev_saved_modules import thread, threading
from _pydev_bundle import pydev_log, pydev_monkey
import os.path
import platform
import ctypes
from io import StringIO
import sys
import traceback
def _load_python_helper_lib_uncached():
    if not IS_CPYTHON or sys.version_info[:2] > (3, 11) or hasattr(sys, 'gettotalrefcount') or (LOAD_NATIVE_LIB_FLAG in ENV_FALSE_LOWER_VALUES):
        pydev_log.info('Helper lib to set tracing to all threads not loaded.')
        return None
    try:
        filename = get_python_helper_lib_filename()
        if filename is None:
            return None
        lib = ctypes.pydll.LoadLibrary(filename)
        pydev_log.info('Successfully Loaded helper lib to set tracing to all threads.')
        return lib
    except:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 1:
            pydev_log.exception('Error loading: %s', filename)
        return None