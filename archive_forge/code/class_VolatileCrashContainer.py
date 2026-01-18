from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
class VolatileCrashContainer(CrashTable):
    """
    Old in-memory crash dump storage.

    @warning:
        Superceded by L{CrashDictionary} since WinAppDbg 1.5.
        New applications should not use this class.
    """

    def __init__(self, allowRepeatedKeys=True):
        """
        Volatile containers are stored only in memory and
        destroyed when they go out of scope.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same key as a
            previously existing object will be ignored.
        """
        super(VolatileCrashContainer, self).__init__(allowRepeatedKeys=allowRepeatedKeys)