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
class ProcessWriter(abstract.FileDescriptor):
    """
    (Internal) Helper class to write into a Process's input pipe.

    I am a helper which describes a selectable asynchronous writer to a
    process's input pipe, including stdin.

    @ivar enableReadHack: A flag which determines how readability on this
        write descriptor will be handled.  If C{True}, then readability may
        indicate the reader for this write descriptor has been closed (ie,
        the connection has been lost).  If C{False}, then readability events
        are ignored.
    """
    connected = 1
    ic = 0
    enableReadHack = False

    def __init__(self, reactor, proc, name, fileno, forceReadHack=False):
        """
        Initialize, specifying a Process instance to connect to.
        """
        abstract.FileDescriptor.__init__(self, reactor)
        fdesc.setNonBlocking(fileno)
        self.proc = proc
        self.name = name
        self.fd = fileno
        if not stat.S_ISFIFO(os.fstat(self.fileno()).st_mode):
            self.enableReadHack = False
        elif forceReadHack:
            self.enableReadHack = True
        else:
            try:
                os.read(self.fileno(), 0)
            except OSError:
                self.enableReadHack = True
        if self.enableReadHack:
            self.startReading()

    def fileno(self):
        """
        Return the fileno() of my process's stdin.
        """
        return self.fd

    def writeSomeData(self, data):
        """
        Write some data to the open process.
        """
        rv = fdesc.writeToFD(self.fd, data)
        if rv == len(data) and self.enableReadHack:
            self.startReading()
        return rv

    def write(self, data):
        self.stopReading()
        abstract.FileDescriptor.write(self, data)

    def doRead(self):
        """
        The only way a write pipe can become "readable" is at EOF, because the
        child has closed it, and we're using a reactor which doesn't
        distinguish between readable and closed (such as the select reactor).

        Except that's not true on linux < 2.6.11. It has the following
        characteristics: write pipe is completely empty => POLLOUT (writable in
        select), write pipe is not completely empty => POLLIN (readable in
        select), write pipe's reader closed => POLLIN|POLLERR (readable and
        writable in select)

        That's what this funky code is for. If linux was not broken, this
        function could be simply "return CONNECTION_LOST".
        """
        if self.enableReadHack:
            return CONNECTION_LOST
        else:
            self.stopReading()

    def connectionLost(self, reason):
        """
        See abstract.FileDescriptor.connectionLost.
        """
        fdesc.setBlocking(self.fd)
        abstract.FileDescriptor.connectionLost(self, reason)
        self.proc.childConnectionLost(self.name, reason)