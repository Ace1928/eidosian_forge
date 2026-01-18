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
class ProcessReader(abstract.FileDescriptor):
    """
    ProcessReader

    I am a selectable representation of a process's output pipe, such as
    stdout and stderr.
    """
    connected = True

    def __init__(self, reactor, proc, name, fileno):
        """
        Initialize, specifying a process to connect to.
        """
        abstract.FileDescriptor.__init__(self, reactor)
        fdesc.setNonBlocking(fileno)
        self.proc = proc
        self.name = name
        self.fd = fileno
        self.startReading()

    def fileno(self):
        """
        Return the fileno() of my process's stderr.
        """
        return self.fd

    def writeSomeData(self, data):
        assert data == b''
        return CONNECTION_LOST

    def doRead(self):
        """
        This is called when the pipe becomes readable.
        """
        return fdesc.readFromFD(self.fd, self.dataReceived)

    def dataReceived(self, data):
        self.proc.childDataReceived(self.name, data)

    def loseConnection(self):
        if self.connected and (not self.disconnecting):
            self.disconnecting = 1
            self.stopReading()
            self.reactor.callLater(0, self.connectionLost, failure.Failure(CONNECTION_DONE))

    def connectionLost(self, reason):
        """
        Close my end of the pipe, signal the Process (which signals the
        ProcessProtocol).
        """
        abstract.FileDescriptor.connectionLost(self, reason)
        self.proc.childConnectionLost(self.name, reason)