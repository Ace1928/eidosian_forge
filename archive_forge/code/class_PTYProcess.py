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
@implementer(IProcessTransport)
class PTYProcess(abstract.FileDescriptor, _BaseProcess):
    """
    An operating-system Process that uses PTY support.
    """
    status = -1
    pid = None

    def __init__(self, reactor, executable, args, environment, path, proto, uid=None, gid=None, usePTY=None):
        """
        Spawn an operating-system process.

        This is where the hard work of disconnecting all currently open
        files / forking / executing the new process happens.  (This is
        executed automatically when a Process is instantiated.)

        This will also run the subprocess as a given user ID and group ID, if
        specified.  (Implementation Note: this doesn't support all the arcane
        nuances of setXXuid on UNIX: it will assume that either your effective
        or real UID is 0.)
        """
        if pty is None and (not isinstance(usePTY, (tuple, list))):
            raise NotImplementedError('cannot use PTYProcess on platforms without the pty module.')
        abstract.FileDescriptor.__init__(self, reactor)
        _BaseProcess.__init__(self, proto)
        if isinstance(usePTY, (tuple, list)):
            masterfd, slavefd, _ = usePTY
        else:
            masterfd, slavefd = pty.openpty()
        try:
            self._fork(path, uid, gid, executable, args, environment, masterfd=masterfd, slavefd=slavefd)
        except BaseException:
            if not isinstance(usePTY, (tuple, list)):
                os.close(masterfd)
                os.close(slavefd)
            raise
        os.close(slavefd)
        fdesc.setNonBlocking(masterfd)
        self.fd = masterfd
        self.startReading()
        self.connected = 1
        self.status = -1
        try:
            self.proto.makeConnection(self)
        except BaseException:
            log.err()
        registerReapProcessHandler(self.pid, self)

    def _setupChild(self, masterfd, slavefd):
        """
        Set up child process after C{fork()} but before C{exec()}.

        This involves:

            - closing C{masterfd}, since it is not used in the subprocess

            - creating a new session with C{os.setsid}

            - changing the controlling terminal of the process (and the new
              session) to point at C{slavefd}

            - duplicating C{slavefd} to standard input, output, and error

            - closing all other open file descriptors (according to
              L{_listOpenFDs})

            - re-setting all signal handlers to C{SIG_DFL}

        @param masterfd: The master end of a PTY file descriptors opened with
            C{openpty}.
        @type masterfd: L{int}

        @param slavefd: The slave end of a PTY opened with C{openpty}.
        @type slavefd: L{int}
        """
        os.close(masterfd)
        os.setsid()
        fcntl.ioctl(slavefd, termios.TIOCSCTTY, '')
        for fd in range(3):
            if fd != slavefd:
                os.close(fd)
        os.dup2(slavefd, 0)
        os.dup2(slavefd, 1)
        os.dup2(slavefd, 2)
        for fd in _listOpenFDs():
            if fd > 2:
                try:
                    os.close(fd)
                except BaseException:
                    pass
        self._resetSignalDisposition()

    def closeStdin(self):
        pass

    def closeStdout(self):
        pass

    def closeStderr(self):
        pass

    def doRead(self):
        """
        Called when my standard output stream is ready for reading.
        """
        return fdesc.readFromFD(self.fd, lambda data: self.proto.childDataReceived(1, data))

    def fileno(self):
        """
        This returns the file number of standard output on this process.
        """
        return self.fd

    def maybeCallProcessEnded(self):
        if self.lostProcess == 2:
            _BaseProcess.maybeCallProcessEnded(self)

    def connectionLost(self, reason):
        """
        I call this to clean up when one or all of my connections has died.
        """
        abstract.FileDescriptor.connectionLost(self, reason)
        os.close(self.fd)
        self.lostProcess += 1
        self.maybeCallProcessEnded()

    def writeSomeData(self, data):
        """
        Write some data to the open process.
        """
        return fdesc.writeToFD(self.fd, data)

    def closeChildFD(self, descriptor):
        raise NotImplementedError()

    def writeToChild(self, childFD, data):
        raise NotImplementedError()