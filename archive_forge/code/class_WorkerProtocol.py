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
class WorkerProtocol(AMP):
    """
    The worker-side trial distributed protocol.
    """
    logger = Logger()

    def __init__(self, forceGarbageCollection=False):
        self._loader = TestLoader()
        self._result = WorkerReporter(self)
        self._forceGarbageCollection = forceGarbageCollection

    @workercommands.Run.responder
    async def run(self, testCase: str) -> RunResult:
        """
        Run a test case by name.
        """
        with self._result.gatherReportingResults() as results:
            case = self._loader.loadByName(testCase)
            suite = TrialSuite([case], self._forceGarbageCollection)
            suite.run(self._result)
        allSucceeded = True
        for success, result in await DeferredList(results, consumeErrors=True):
            if success:
                continue
            allSucceeded = False
            self.logger.failure('Result reporting for {id} failed', failure=result, id=testCase)
            try:
                await self._result.addErrorFallible(testCase, result)
            except BaseException:
                self.logger.failure('Additionally, reporting the reporting failure failed.')
        return {'success': allSucceeded}

    @workercommands.Start.responder
    def start(self, directory):
        """
        Set up the worker, moving into given directory for tests to run in
        them.
        """
        os.chdir(directory)
        return {'success': True}