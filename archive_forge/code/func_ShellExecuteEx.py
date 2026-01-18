from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def ShellExecuteEx(lpExecInfo):
    if isinstance(lpExecInfo, SHELLEXECUTEINFOA):
        ShellExecuteExA(lpExecInfo)
    elif isinstance(lpExecInfo, SHELLEXECUTEINFOW):
        ShellExecuteExW(lpExecInfo)
    else:
        raise TypeError('Expected SHELLEXECUTEINFOA or SHELLEXECUTEINFOW, got %s instead' % type(lpExecInfo))