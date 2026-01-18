from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
def _sanitize_path(self, path):
    """
        Sanitizes the given Registry path.

        @type  path: str
        @param path: Registry path.

        @rtype:  str
        @return: Registry path.
        """
    return self._join_path(*self._split_path(path))