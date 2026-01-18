import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DeleteProcThreadAttributeList(lpAttributeList):
    _DeleteProcThreadAttributeList = windll.kernel32.DeleteProcThreadAttributeList
    _DeleteProcThreadAttributeList.restype = None
    _DeleteProcThreadAttributeList(byref(lpAttributeList))