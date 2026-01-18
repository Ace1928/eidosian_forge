from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
def _split_path(self, path):
    """
        Splits a Registry path and returns the hive and key.

        @type  path: str
        @param path: Registry path.

        @rtype:  tuple( int, str )
        @return: Tuple containing the hive handle and the subkey path.
            The hive handle is always one of the following integer constants:
             - L{win32.HKEY_CLASSES_ROOT}
             - L{win32.HKEY_CURRENT_USER}
             - L{win32.HKEY_LOCAL_MACHINE}
             - L{win32.HKEY_USERS}
             - L{win32.HKEY_PERFORMANCE_DATA}
             - L{win32.HKEY_CURRENT_CONFIG}
        """
    if '\\' in path:
        p = path.find('\\')
        hive = path[:p]
        path = path[p + 1:]
    else:
        hive = path
        path = None
    handle = self._hives_by_name[hive.upper()]
    return (handle, path)