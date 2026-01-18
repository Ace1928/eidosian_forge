from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def LookupAccountSidW(lpSystemName, lpSid):
    _LookupAccountSidW = windll.advapi32.LookupAccountSidA
    _LookupAccountSidW.argtypes = [LPSTR, PSID, LPWSTR, LPDWORD, LPWSTR, LPDWORD, LPDWORD]
    _LookupAccountSidW.restype = bool
    cchName = DWORD(0)
    cchReferencedDomainName = DWORD(0)
    peUse = DWORD(0)
    _LookupAccountSidW(lpSystemName, lpSid, None, byref(cchName), None, byref(cchReferencedDomainName), byref(peUse))
    error = GetLastError()
    if error != ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(error)
    lpName = ctypes.create_unicode_buffer(u'', cchName + 1)
    lpReferencedDomainName = ctypes.create_unicode_buffer(u'', cchReferencedDomainName + 1)
    success = _LookupAccountSidW(lpSystemName, lpSid, lpName, byref(cchName), lpReferencedDomainName, byref(cchReferencedDomainName), byref(peUse))
    if not success:
        raise ctypes.WinError()
    return (lpName.value, lpReferencedDomainName.value, peUse.value)