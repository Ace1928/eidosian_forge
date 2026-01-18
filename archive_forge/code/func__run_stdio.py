import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, \
def _run_stdio(self):
    """Runs the process using the system standard I/O.

        IMPORTANT: stdin needs to be asynchronous, so the Python
                   sys.stdin object is not used. Instead,
                   msvcrt.kbhit/getwch are used asynchronously.
        """
    if self.mergeout:
        return self.run(stdout_func=self._stdout_raw, stdin_func=self._stdin_raw_block)
    else:
        return self.run(stdout_func=self._stdout_raw, stdin_func=self._stdin_raw_block, stderr_func=self._stderr_raw)