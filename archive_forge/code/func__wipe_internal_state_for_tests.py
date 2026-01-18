import atexit
import contextlib
import sys
from .ansitowin32 import AnsiToWin32
def _wipe_internal_state_for_tests():
    global orig_stdout, orig_stderr
    orig_stdout = None
    orig_stderr = None
    global wrapped_stdout, wrapped_stderr
    wrapped_stdout = None
    wrapped_stderr = None
    global atexit_done
    atexit_done = False
    global fixed_windows_console
    fixed_windows_console = False
    try:
        atexit.unregister(reset_all)
    except AttributeError:
        pass