from __future__ import annotations
import sys
from zope.interface import implementer
from CFNetwork import (
from CoreFoundation import (
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, PosixReactorBase
from twisted.python import log
from ._signals import _UnixWaker
def _watchFD(self, fd, descr, flag):
    """
        Register a file descriptor with the C{CFRunLoop}, or modify its state
        so that it's listening for both notifications (read and write) rather
        than just one; used to implement C{addReader} and C{addWriter}.

        @param fd: The file descriptor.

        @type fd: L{int}

        @param descr: the L{IReadDescriptor} or L{IWriteDescriptor}

        @param flag: the flag to register for callbacks on, either
            C{kCFSocketReadCallBack} or C{kCFSocketWriteCallBack}
        """
    if fd == -1:
        raise RuntimeError('Invalid file descriptor.')
    if fd in self._fdmap:
        src, cfs, gotdescr, rw = self._fdmap[fd]
    else:
        ctx = []
        ctx.append(fd)
        cfs = CFSocketCreateWithNative(kCFAllocatorDefault, fd, kCFSocketReadCallBack | kCFSocketWriteCallBack | kCFSocketConnectCallBack, self._socketCallback, ctx)
        CFSocketSetSocketFlags(cfs, kCFSocketAutomaticallyReenableReadCallBack | kCFSocketAutomaticallyReenableWriteCallBack | _preserveSOError)
        src = CFSocketCreateRunLoopSource(kCFAllocatorDefault, cfs, 0)
        ctx.append(src)
        CFRunLoopAddSource(self._cfrunloop, src, kCFRunLoopCommonModes)
        CFSocketDisableCallBacks(cfs, kCFSocketReadCallBack | kCFSocketWriteCallBack | kCFSocketConnectCallBack)
        rw = [False, False]
        self._idmap[id(descr)] = fd
        self._fdmap[fd] = (src, cfs, descr, rw)
    rw[self._flag2idx(flag)] = True
    CFSocketEnableCallBacks(cfs, flag)