import sys
from os_win._i18n import _
class ClusterWin32Exception(ClusterException, Win32Exception):
    pass