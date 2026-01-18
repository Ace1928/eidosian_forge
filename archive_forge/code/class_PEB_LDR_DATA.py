from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class PEB_LDR_DATA(Structure):
    _fields_ = [('Length', ULONG), ('Initialized', BOOLEAN), ('SsHandle', PVOID), ('InLoadOrderModuleList', LIST_ENTRY), ('InMemoryOrderModuleList', LIST_ENTRY), ('InInitializationOrderModuleList', LIST_ENTRY)]