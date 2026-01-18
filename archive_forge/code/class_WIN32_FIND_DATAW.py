import ctypes
class WIN32_FIND_DATAW(ctypes.Structure):
    _fields_ = [('dwFileAttributes', DWORD), ('ftCreationTime', FILETIME), ('ftLastAccessTime', FILETIME), ('ftLastWriteTime', FILETIME), ('nFileSizeHigh', DWORD), ('nFileSizeLow', DWORD), ('dwReserved0', DWORD), ('dwReserved1', DWORD), ('cFileName', WCHAR * MAX_PATH), ('cAlternateFileName', WCHAR * 14)]