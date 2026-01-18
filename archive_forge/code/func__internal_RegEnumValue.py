from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def _internal_RegEnumValue(ansi, hKey, dwIndex, bGetData=True):
    if ansi:
        _RegEnumValue = windll.advapi32.RegEnumValueA
        _RegEnumValue.argtypes = [HKEY, DWORD, LPSTR, LPDWORD, LPVOID, LPDWORD, LPVOID, LPDWORD]
    else:
        _RegEnumValue = windll.advapi32.RegEnumValueW
        _RegEnumValue.argtypes = [HKEY, DWORD, LPWSTR, LPDWORD, LPVOID, LPDWORD, LPVOID, LPDWORD]
    _RegEnumValue.restype = LONG
    cchValueName = DWORD(1024)
    dwType = DWORD(-1)
    lpcchValueName = byref(cchValueName)
    lpType = byref(dwType)
    if ansi:
        lpValueName = ctypes.create_string_buffer(cchValueName.value)
    else:
        lpValueName = ctypes.create_unicode_buffer(cchValueName.value)
    if bGetData:
        cbData = DWORD(0)
        lpcbData = byref(cbData)
    else:
        lpcbData = None
    lpData = None
    errcode = _RegEnumValue(hKey, dwIndex, lpValueName, lpcchValueName, None, lpType, lpData, lpcbData)
    if errcode == ERROR_MORE_DATA or (bGetData and errcode == ERROR_SUCCESS):
        if ansi:
            cchValueName.value = cchValueName.value + sizeof(CHAR)
            lpValueName = ctypes.create_string_buffer(cchValueName.value)
        else:
            cchValueName.value = cchValueName.value + sizeof(WCHAR)
            lpValueName = ctypes.create_unicode_buffer(cchValueName.value)
        if bGetData:
            Type = dwType.value
            if Type in (REG_DWORD, REG_DWORD_BIG_ENDIAN):
                if cbData.value != sizeof(DWORD):
                    raise ValueError('REG_DWORD value of size %d' % cbData.value)
                Data = DWORD(0)
            elif Type == REG_QWORD:
                if cbData.value != sizeof(QWORD):
                    raise ValueError('REG_QWORD value of size %d' % cbData.value)
                Data = QWORD(long(0))
            elif Type in (REG_SZ, REG_EXPAND_SZ, REG_MULTI_SZ):
                if ansi:
                    Data = ctypes.create_string_buffer(cbData.value)
                else:
                    Data = ctypes.create_unicode_buffer(cbData.value)
            elif Type == REG_LINK:
                Data = ctypes.create_unicode_buffer(cbData.value)
            else:
                Data = ctypes.create_string_buffer(cbData.value)
            lpData = byref(Data)
        errcode = _RegEnumValue(hKey, dwIndex, lpValueName, lpcchValueName, None, lpType, lpData, lpcbData)
    if errcode == ERROR_NO_MORE_ITEMS:
        return None
    if not bGetData:
        return (lpValueName.value, dwType.value)
    if Type in (REG_DWORD, REG_DWORD_BIG_ENDIAN, REG_QWORD, REG_SZ, REG_EXPAND_SZ, REG_LINK):
        return (lpValueName.value, dwType.value, Data.value)
    if Type == REG_MULTI_SZ:
        sData = Data[:]
        del Data
        if ansi:
            aData = sData.split('\x00')
        else:
            aData = sData.split(u'\x00')
        aData = [token for token in aData if token]
        return (lpValueName.value, dwType.value, aData)
    return (lpValueName.value, dwType.value, Data.raw)