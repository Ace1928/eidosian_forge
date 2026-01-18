from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def __set_Pointer(self, ptr):
    self.Status = ptr.value