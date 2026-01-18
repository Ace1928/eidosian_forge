from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
class IO_STATUS_BLOCK(Structure):
    _fields_ = [('Status', NTSTATUS), ('Information', ULONG_PTR)]

    def __get_Pointer(self):
        return PVOID(self.Status)

    def __set_Pointer(self, ptr):
        self.Status = ptr.value
    Pointer = property(__get_Pointer, __set_Pointer)