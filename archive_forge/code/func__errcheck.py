import ctypes
from ctypes.wintypes import BOOL, DWORD, HANDLE, LARGE_INTEGER, LPCSTR, UINT
from debugpy.common import log
def _errcheck(is_error_result=lambda result: not result):

    def impl(result, func, args):
        if is_error_result(result):
            log.debug('{0} returned {1}', func.__name__, result)
            raise ctypes.WinError()
        else:
            return result
    return impl