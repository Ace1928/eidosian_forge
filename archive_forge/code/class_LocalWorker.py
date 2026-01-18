import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, TextIO, TypeVar
from unittest import TestCase
from zope.interface import implementer
from attrs import frozen
from typing_extensions import Protocol, TypedDict
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.internet.protocol import ProcessProtocol
from twisted.logger import Logger
from twisted.protocols.amp import AMP
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedObject
from twisted.trial._dist import (
from twisted.trial._dist.workerreporter import WorkerReporter
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader, TrialSuite
from twisted.trial.unittest import Todo
from .stream import StreamOpen, StreamReceiver, StreamWrite
class LocalWorker(ProcessProtocol):
    """
    Local process worker protocol. This worker runs as a local process and
    communicates via stdin/out.

    @ivar _ampProtocol: The L{AMP} protocol instance used to communicate with
        the worker.

    @ivar _logDirectory: The directory where logs will reside.

    @ivar _logFile: The main log file for tests output.
    """

    def __init__(self, ampProtocol: LocalWorkerAMP, logDirectory: FilePath[Any], logFile: TextIO):
        self._ampProtocol = ampProtocol
        self._logDirectory = logDirectory
        self._logFile = logFile
        self.endDeferred: Deferred[None] = Deferred()

    async def exit(self) -> None:
        """
        Cause the worker process to exit.
        """
        if self.transport is None:
            raise NotRunning()
        endDeferred = self.endDeferred
        self.transport.closeChildFD(_WORKER_AMP_STDIN)
        try:
            await endDeferred
        except ProcessDone:
            pass

    def connectionMade(self):
        """
        When connection is made, create the AMP protocol instance.
        """
        self._ampProtocol.makeConnection(LocalWorkerTransport(self.transport))
        self._logDirectory.makedirs(ignoreExistingDirectory=True)
        self._outLog = self._logDirectory.child('out.log').open('w')
        self._errLog = self._logDirectory.child('err.log').open('w')
        self._ampProtocol.setTestStream(self._logFile)
        d = self._ampProtocol.callRemote(workercommands.Start, directory=self._logDirectory.path)
        d.addErrback(lambda x: None)

    def connectionLost(self, reason):
        """
        On connection lost, close the log files that we're managing for stdin
        and stdout.
        """
        self._outLog.close()
        self._errLog.close()
        self.transport = None

    def processEnded(self, reason: Failure) -> None:
        """
        When the process closes, call C{connectionLost} for cleanup purposes
        and forward the information to the C{_ampProtocol}.
        """
        self.connectionLost(reason)
        self._ampProtocol.connectionLost(reason)
        self.endDeferred.callback(reason)

    def outReceived(self, data):
        """
        Send data received from stdout to log.
        """
        self._outLog.write(data)

    def errReceived(self, data):
        """
        Write error data to log.
        """
        self._errLog.write(data)

    def childDataReceived(self, childFD, data):
        """
        Handle data received on the specific pipe for the C{_ampProtocol}.
        """
        if childFD == _WORKER_AMP_STDOUT:
            self._ampProtocol.dataReceived(data)
        else:
            ProcessProtocol.childDataReceived(self, childFD, data)