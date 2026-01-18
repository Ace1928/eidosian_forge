import ctypes
import ctypes.wintypes
import stat as stdstat
from collections import namedtuple
def _to_unix_time(ft):
    t = ft.dwHighDateTime << 32 | ft.dwLowDateTime
    return t / 10000000 - 11644473600