from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def __filename_to_modname(self, pathname):
    """
        @type  pathname: str
        @param pathname: Pathname to a module.

        @rtype:  str
        @return: Module name.
        """
    filename = PathOperations.pathname_to_filename(pathname)
    if filename:
        filename = filename.lower()
        filepart, extpart = PathOperations.split_extension(filename)
        if filepart and extpart:
            modName = filepart
        else:
            modName = filename
    else:
        modName = pathname
    return modName