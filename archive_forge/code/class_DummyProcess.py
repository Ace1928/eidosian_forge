import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
class DummyProcess:
    """
    An incomplete and fake L{IProcessTransport} implementation for testing how
    L{ProcessMonitor} behaves when its monitored processes exit.

    @ivar _terminationDelay: the delay in seconds after which the DummyProcess
        will appear to exit when it receives a TERM signal
    """
    pid = 1
    proto = None
    _terminationDelay = 1

    def __init__(self, reactor, executable, args, environment, path, proto, uid=None, gid=None, usePTY=0, childFDs=None):
        self.proto = proto
        self._reactor = reactor
        self._executable = executable
        self._args = args
        self._environment = environment
        self._path = path
        self._uid = uid
        self._gid = gid
        self._usePTY = usePTY
        self._childFDs = childFDs

    def signalProcess(self, signalID):
        """
        A partial implementation of signalProcess which can only handle TERM and
        KILL signals.
         - When a TERM signal is given, the dummy process will appear to exit
           after L{DummyProcess._terminationDelay} seconds with exit code 0
         - When a KILL signal is given, the dummy process will appear to exit
           immediately with exit code 1.

        @param signalID: The signal name or number to be issued to the process.
        @type signalID: C{str}
        """
        params = {'TERM': (self._terminationDelay, 0), 'KILL': (0, 1)}
        if self.pid is None:
            raise ProcessExitedAlready()
        if signalID in params:
            delay, status = params[signalID]
            self._signalHandler = self._reactor.callLater(delay, self.processEnded, status)

    def processEnded(self, status):
        """
        Deliver the process ended event to C{self.proto}.
        """
        self.pid = None
        statusMap = {0: ProcessDone, 1: ProcessTerminated}
        self.proto.processEnded(Failure(statusMap[status](status)))