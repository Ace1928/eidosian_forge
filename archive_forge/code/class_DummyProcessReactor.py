import pickle
from twisted.internet.error import ProcessDone, ProcessExitedAlready, ProcessTerminated
from twisted.internet.task import Clock
from twisted.internet.testing import MemoryReactor
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.runner.procmon import LoggingProtocol, ProcessMonitor
from twisted.trial import unittest
class DummyProcessReactor(MemoryReactor, Clock):
    """
    @ivar spawnedProcesses: a list that keeps track of the fake process
        instances built by C{spawnProcess}.
    @type spawnedProcesses: C{list}
    """

    def __init__(self):
        MemoryReactor.__init__(self)
        Clock.__init__(self)
        self.spawnedProcesses = []

    def spawnProcess(self, processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        """
        Fake L{reactor.spawnProcess}, that logs all the process
        arguments and returns a L{DummyProcess}.
        """
        proc = DummyProcess(self, executable, args, env, path, processProtocol, uid, gid, usePTY, childFDs)
        processProtocol.makeConnection(proc)
        self.spawnedProcesses.append(proc)
        return proc