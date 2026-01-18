from __future__ import annotations
import errno
import gc
import io
import os
import signal
import stat
import sys
import traceback
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from zope.interface import implementer
from twisted.internet import abstract, error, fdesc
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IProcessTransport
from twisted.internet.main import CONNECTION_DONE, CONNECTION_LOST
from twisted.python import failure, log
from twisted.python.runtime import platform
from twisted.python.util import switchUID
class _FDDetector:
    """
    This class contains the logic necessary to decide which of the available
    system techniques should be used to detect the open file descriptors for
    the current process. The chosen technique gets monkey-patched into the
    _listOpenFDs method of this class so that the detection only needs to occur
    once.

    @ivar listdir: The implementation of listdir to use. This gets overwritten
        by the test cases.
    @ivar getpid: The implementation of getpid to use, returns the PID of the
        running process.
    @ivar openfile: The implementation of open() to use, by default the Python
        builtin.
    """
    listdir = os.listdir
    getpid = os.getpid
    openfile = open

    def __init__(self):
        self._implementations = [self._procFDImplementation, self._devFDImplementation, self._fallbackFDImplementation]

    def _listOpenFDs(self):
        """
        Return an iterable of file descriptors which I{may} be open in this
        process.

        This will try to return the fewest possible descriptors without missing
        any.
        """
        self._listOpenFDs = self._getImplementation()
        return self._listOpenFDs()

    def _getImplementation(self):
        """
        Pick a method which gives correct results for C{_listOpenFDs} in this
        runtime environment.

        This involves a lot of very platform-specific checks, some of which may
        be relatively expensive.  Therefore the returned method should be saved
        and re-used, rather than always calling this method to determine what it
        is.

        See the implementation for the details of how a method is selected.
        """
        for impl in self._implementations:
            try:
                before = impl()
            except BaseException:
                continue
            with self.openfile('/dev/null', 'r'):
                after = impl()
            if before != after:
                return impl
        return impl

    def _devFDImplementation(self):
        """
        Simple implementation for systems where /dev/fd actually works.
        See: http://www.freebsd.org/cgi/man.cgi?fdescfs
        """
        dname = '/dev/fd'
        result = [int(fd) for fd in self.listdir(dname)]
        return result

    def _procFDImplementation(self):
        """
        Simple implementation for systems where /proc/pid/fd exists (we assume
        it works).
        """
        dname = '/proc/%d/fd' % (self.getpid(),)
        return [int(fd) for fd in self.listdir(dname)]

    def _fallbackFDImplementation(self):
        """
        Fallback implementation where either the resource module can inform us
        about the upper bound of how many FDs to expect, or where we just guess
        a constant maximum if there is no resource module.

        All possible file descriptors from 0 to that upper bound are returned
        with no attempt to exclude invalid file descriptor values.
        """
        try:
            import resource
        except ImportError:
            maxfds = 1024
        else:
            maxfds = min(1024, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
        return range(maxfds)