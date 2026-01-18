from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
def _join_path(self, hive, subkey):
    """
        Joins the hive and key to make a Registry path.

        @type  hive: int
        @param hive: Registry hive handle.
            The hive handle must be one of the following integer constants:
             - L{win32.HKEY_CLASSES_ROOT}
             - L{win32.HKEY_CURRENT_USER}
             - L{win32.HKEY_LOCAL_MACHINE}
             - L{win32.HKEY_USERS}
             - L{win32.HKEY_PERFORMANCE_DATA}
             - L{win32.HKEY_CURRENT_CONFIG}

        @type  subkey: str
        @param subkey: Subkey path.

        @rtype:  str
        @return: Registry path.
        """
    path = self._hives_by_value[hive]
    if subkey:
        path = path + '\\' + subkey
    return path