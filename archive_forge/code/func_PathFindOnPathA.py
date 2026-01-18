from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindOnPathA(pszFile, ppszOtherDirs=None):
    _PathFindOnPathA = windll.shlwapi.PathFindOnPathA
    _PathFindOnPathA.argtypes = [LPSTR, LPSTR]
    _PathFindOnPathA.restype = bool
    pszFile = ctypes.create_string_buffer(pszFile, MAX_PATH)
    if not ppszOtherDirs:
        ppszOtherDirs = None
    else:
        szArray = ''
        for pszOtherDirs in ppszOtherDirs:
            if pszOtherDirs:
                szArray = '%s%s\x00' % (szArray, pszOtherDirs)
        szArray = szArray + '\x00'
        pszOtherDirs = ctypes.create_string_buffer(szArray)
        ppszOtherDirs = ctypes.pointer(pszOtherDirs)
    if _PathFindOnPathA(pszFile, ppszOtherDirs):
        return pszFile.value
    return None