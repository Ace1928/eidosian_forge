import ctypes
import ctypes.wintypes
import stat as stdstat
from collections import namedtuple
def _to_mode(attr):
    m = 0
    if attr & FILE_ATTRIBUTE_DIRECTORY:
        m |= stdstat.S_IFDIR | 73
    else:
        m |= stdstat.S_IFREG
    if attr & FILE_ATTRIBUTE_READONLY:
        m |= 292
    else:
        m |= 438
    return m