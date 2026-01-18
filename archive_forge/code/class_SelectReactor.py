import select
import sys
from errno import EBADF, EINTR
from time import sleep
from typing import Type
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
from twisted.python.runtime import platformType
@implementer(IReactorFDSet)
class SelectReactor(posixbase.PosixReactorBase, _extraBase):
    """
    A select() based reactor - runs on all POSIX platforms and on Win32.

    @ivar _reads: A set containing L{FileDescriptor} instances which will be
        checked for read events.

    @ivar _writes: A set containing L{FileDescriptor} instances which will be
        checked for writability.
    """

    def __init__(self):
        """
        Initialize file descriptor tracking dictionaries and the base class.
        """
        self._reads = set()
        self._writes = set()
        posixbase.PosixReactorBase.__init__(self)

    def _preenDescriptors(self):
        log.msg('Malformed file descriptor found.  Preening lists.')
        readers = list(self._reads)
        writers = list(self._writes)
        self._reads.clear()
        self._writes.clear()
        for selSet, selList in ((self._reads, readers), (self._writes, writers)):
            for selectable in selList:
                try:
                    select.select([selectable], [selectable], [selectable], 0)
                except Exception as e:
                    log.msg('bad descriptor %s' % selectable)
                    self._disconnectSelectable(selectable, e, False)
                else:
                    selSet.add(selectable)

    def doSelect(self, timeout):
        """
        Run one iteration of the I/O monitor loop.

        This will run all selectables who had input or output readiness
        waiting for them.
        """
        try:
            r, w, ignored = _select(self._reads, self._writes, [], timeout)
        except ValueError:
            self._preenDescriptors()
            return
        except TypeError:
            log.err()
            self._preenDescriptors()
            return
        except OSError as se:
            if se.args[0] in (0, 2):
                if not self._reads and (not self._writes):
                    return
                else:
                    raise
            elif se.args[0] == EINTR:
                return
            elif se.args[0] == EBADF:
                self._preenDescriptors()
                return
            else:
                raise
        _drdw = self._doReadOrWrite
        _logrun = log.callWithLogger
        for selectables, method, fdset in ((r, 'doRead', self._reads), (w, 'doWrite', self._writes)):
            for selectable in selectables:
                if selectable not in fdset:
                    continue
                _logrun(selectable, _drdw, selectable, method)
    doIteration = doSelect

    def _doReadOrWrite(self, selectable, method):
        try:
            why = getattr(selectable, method)()
        except BaseException:
            why = sys.exc_info()[1]
            log.err()
        if why:
            self._disconnectSelectable(selectable, why, method == 'doRead')

    def addReader(self, reader):
        """
        Add a FileDescriptor for notification of data available to read.
        """
        self._reads.add(reader)

    def addWriter(self, writer):
        """
        Add a FileDescriptor for notification of data available to write.
        """
        self._writes.add(writer)

    def removeReader(self, reader):
        """
        Remove a Selectable for notification of data available to read.
        """
        self._reads.discard(reader)

    def removeWriter(self, writer):
        """
        Remove a Selectable for notification of data available to write.
        """
        self._writes.discard(writer)

    def removeAll(self):
        return self._removeAll(self._reads, self._writes)

    def getReaders(self):
        return list(self._reads)

    def getWriters(self):
        return list(self._writes)