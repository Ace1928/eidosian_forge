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
class LocalWorkerAMP(AMP):
    """
    Local implementation of the manager commands.
    """

    def __init__(self, boxReceiver=None, locator=None):
        super().__init__(boxReceiver, locator)
        self._streams = StreamReceiver()

    @StreamOpen.responder
    def streamOpen(self):
        return {'streamId': self._streams.open()}

    @StreamWrite.responder
    def streamWrite(self, streamId, data):
        self._streams.write(streamId, data)
        return {}

    @managercommands.AddSuccess.responder
    def addSuccess(self, testName):
        """
        Add a success to the reporter.
        """
        self._result.addSuccess(self._testCase)
        return {'success': True}

    def _buildFailure(self, error: WorkerException, errorClass: str, frames: List[str]) -> Failure:
        """
        Helper to build a C{Failure} with some traceback.

        @param error: An C{Exception} instance.

        @param errorClass: The class name of the C{error} class.

        @param frames: A flat list of strings representing the information need
            to approximatively rebuild C{Failure} frames.

        @return: A L{Failure} instance with enough information about a test
           error.
        """
        errorType = namedObject(errorClass)
        failure = Failure(error, errorType)
        for i in range(0, len(frames), 3):
            failure.frames.append((frames[i], frames[i + 1], int(frames[i + 2]), [], []))
        return failure

    @managercommands.AddError.responder
    def addError(self, testName: str, errorClass: str, errorStreamId: int, framesStreamId: int) -> Dict[str, bool]:
        """
        Add an error to the reporter.

        @param errorStreamId: The identifier of a stream over which the text
            of this error was previously completely sent to the peer.

        @param framesStreamId: The identifier of a stream over which the lines
            of the traceback for this error were previously completely sent to
            the peer.

        @param error: A message describing the error.
        """
        error = b''.join(self._streams.finish(errorStreamId)).decode('utf-8')
        frames = [frame.decode('utf-8') for frame in self._streams.finish(framesStreamId)]
        failure = self._buildFailure(WorkerException(error), errorClass, frames)
        self._result.addError(self._testCase, failure)
        return {'success': True}

    @managercommands.AddFailure.responder
    def addFailure(self, testName: str, failStreamId: int, failClass: str, framesStreamId: int) -> Dict[str, bool]:
        """
        Add a failure to the reporter.

        @param failStreamId: The identifier of a stream over which the text of
            this failure was previously completely sent to the peer.

        @param framesStreamId: The identifier of a stream over which the lines
            of the traceback for this error were previously completely sent to the
            peer.
        """
        fail = b''.join(self._streams.finish(failStreamId)).decode('utf-8')
        frames = [frame.decode('utf-8') for frame in self._streams.finish(framesStreamId)]
        failure = self._buildFailure(WorkerException(fail), failClass, frames)
        self._result.addFailure(self._testCase, failure)
        return {'success': True}

    @managercommands.AddSkip.responder
    def addSkip(self, testName, reason):
        """
        Add a skip to the reporter.
        """
        self._result.addSkip(self._testCase, reason)
        return {'success': True}

    @managercommands.AddExpectedFailure.responder
    def addExpectedFailure(self, testName: str, errorStreamId: int, todo: Optional[str]) -> Dict[str, bool]:
        """
        Add an expected failure to the reporter.

        @param errorStreamId: The identifier of a stream over which the text
            of this error was previously completely sent to the peer.
        """
        error = b''.join(self._streams.finish(errorStreamId)).decode('utf-8')
        _todo = Todo('<unknown>' if todo is None else todo)
        self._result.addExpectedFailure(self._testCase, error, _todo)
        return {'success': True}

    @managercommands.AddUnexpectedSuccess.responder
    def addUnexpectedSuccess(self, testName, todo):
        """
        Add an unexpected success to the reporter.
        """
        self._result.addUnexpectedSuccess(self._testCase, todo)
        return {'success': True}

    @managercommands.TestWrite.responder
    def testWrite(self, out):
        """
        Print test output from the worker.
        """
        self._testStream.write(out + '\n')
        self._testStream.flush()
        return {'success': True}

    async def run(self, testCase: TestCase, result: TestResult) -> RunResult:
        """
        Run a test.
        """
        self._testCase = testCase
        self._result = result
        self._result.startTest(testCase)
        testCaseId = testCase.id()
        try:
            return await self.callRemote(workercommands.Run, testCase=testCaseId)
        finally:
            self._result.stopTest(testCase)

    def setTestStream(self, stream):
        """
        Set the stream used to log output from tests.
        """
        self._testStream = stream