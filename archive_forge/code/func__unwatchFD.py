from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def _unwatchFD(self, fd, descr, flag):
    """
        Unregister a file descriptor with the C{CFRunLoop}, or modify its state
        so that it's listening for only one notification (read or write) as
        opposed to both; used to implement C{removeReader} and C{removeWriter}.

        @param fd: a file descriptor

        @type fd: C{int}

        @param descr: an L{IReadDescriptor} or L{IWriteDescriptor}

        @param flag: C{kCFSocketWriteCallBack} C{kCFSocketReadCallBack}
        """
    if id(descr) not in self._idmap:
        return
    if fd == -1:
        realfd = self._idmap[id(descr)]
    else:
        realfd = fd
    src, cfs, descr, rw = self._fdmap[realfd]
    CFSocketDisableCallBacks(cfs, flag)
    rw[self._flag2idx(flag)] = False
    if not rw[_READ] and (not rw[_WRITE]):
        del self._idmap[id(descr)]
        del self._fdmap[realfd]
        CFRunLoopRemoveSource(self._cfrunloop, src, kCFRunLoopCommonModes)
        CFSocketInvalidate(cfs)