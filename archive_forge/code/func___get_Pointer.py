from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def __get_Pointer(self):
    return PVOID(self.Status)