import ctypes
class WIN32_FIND_DATAA(ctypes.Structure):
    _fields_ = [('dwFileAttributes', DWORD), ('ftCreationTime', FILETIME), ('ftLastAccessTime', FILETIME), ('ftLastWriteTime', FILETIME), ('nFileSizeHigh', DWORD), ('nFileSizeLow', DWORD), ('dwReserved0', DWORD), ('dwReserved1', DWORD), ('cFileName', CHAR * MAX_PATH), ('cAlternateFileName', CHAR * 14)]