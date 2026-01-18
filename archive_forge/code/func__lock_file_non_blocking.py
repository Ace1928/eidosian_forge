import errno
import threading
from time import sleep
import weakref
def _lock_file_non_blocking(file_):
    res = _WinAPI_LockFile(msvcrt.get_osfhandle(file_.fileno()), 0, 0, 1, 0)
    if res:
        return True
    else:
        err = ctypes.get_last_error()
        if err != 33:
            raise ctypes.WinError(err)
        return False