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
class CrashDictionary(object):
    """
    Dictionary-like persistence interface for L{Crash} objects.

    Currently the only implementation is through L{sql.CrashDAO}.
    """

    def __init__(self, url, creator=None, allowRepeatedKeys=True):
        """
        @type  url: str
        @param url: Connection URL of the crash database.
            See L{sql.CrashDAO.__init__} for more details.

        @type  creator: callable
        @param creator: (Optional) Callback function that creates the SQL
            database connection.

            Normally it's not necessary to use this argument. However in some
            odd cases you may need to customize the database connection, for
            example when using the integrated authentication in MSSQL.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same signature as a
            previously existing object will be ignored.
        """
        global sql
        if sql is None:
            from winappdbg import sql
        self._allowRepeatedKeys = allowRepeatedKeys
        self._dao = sql.CrashDAO(url, creator)

    def add(self, crash):
        """
        Adds a new crash to the container.

        @note:
            When the C{allowRepeatedKeys} parameter of the constructor
            is set to C{False}, duplicated crashes are ignored.

        @see: L{Crash.key}

        @type  crash: L{Crash}
        @param crash: Crash object to add.
        """
        self._dao.add(crash, self._allowRepeatedKeys)

    def get(self, key):
        """
        Retrieves a crash from the container.

        @type  key: L{Crash} signature.
        @param key: Heuristic signature of the crash to get.

        @rtype:  L{Crash} object.
        @return: Crash matching the given signature. If more than one is found,
            retrieve the newest one.

        @see:     L{iterkeys}
        @warning: A B{copy} of each object is returned,
            so any changes made to them will be lost.

            To preserve changes do the following:
                1. Keep a reference to the object.
                2. Delete the object from the set.
                3. Modify the object and add it again.
        """
        found = self._dao.find(signature=key, limit=1, order=-1)
        if not found:
            raise KeyError(key)
        return found[0]

    def __iter__(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.
        """
        offset = 0
        limit = 10
        while 1:
            found = self._dao.find(offset=offset, limit=limit)
            if not found:
                break
            offset += len(found)
            for crash in found:
                yield crash

    def itervalues(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} objects.
        """
        return self.__iter__()

    def iterkeys(self):
        """
        @rtype:  iterator
        @return: Iterator of the contained L{Crash} heuristic signatures.
        """
        for crash in self:
            yield crash.signature

    def __contains__(self, crash):
        """
        @type  crash: L{Crash}
        @param crash: Crash object.

        @rtype:  bool
        @return: C{True} if the Crash object is in the container.
        """
        return self._dao.count(signature=crash.signature) > 0

    def has_key(self, key):
        """
        @type  key: L{Crash} signature.
        @param key: Heuristic signature of the crash to get.

        @rtype:  bool
        @return: C{True} if a matching L{Crash} object is in the container.
        """
        return self._dao.count(signature=key) > 0

    def __len__(self):
        """
        @rtype:  int
        @return: Count of L{Crash} elements in the container.
        """
        return self._dao.count()

    def __bool__(self):
        """
        @rtype:  bool
        @return: C{False} if the container is empty.
        """
        return bool(len(self))