from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def __set_hMonitor(self, hMonitor):
    self.hIcon = hMonitor