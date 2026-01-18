import os
from io import BytesIO, StringIO
from typing import Type
from unittest import TestCase as PyUnitTestCase
from zope.interface.verify import verifyObject
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost, ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist import managercommands
from twisted.trial._dist.worker import (
from twisted.trial.reporter import TestResult
from twisted.trial.test import pyunitcases, skipping
from twisted.trial.unittest import TestCase, makeTodo
from .matchers import isFailure, matches_result, similarFrame
class WorkerProtocolErrorTests(TestCase):
    """
    Tests for L{WorkerProtocol}'s handling of certain errors related to
    running the tests themselves (i.e., not test errors but test
    infrastructure/runner errors).
    """

    def _runErrorTest(self, brokenTestName: str, loggedExceptionType: Type[BaseException]) -> None:
        worker, server, pump = connectedServerAndClient(LocalWorkerAMP, WorkerProtocol, greet=False)
        expectedCase = pyunitcases.BrokenRunInfrastructure(brokenTestName)
        result = TestResult()
        Deferred.fromCoroutine(server.run(expectedCase, result))
        pump.flush()
        assert_that(result, matches_result(errors=has_length(1)))
        [(actualCase, errors)] = result.errors
        assert_that(actualCase, equal_to(expectedCase))
        assert_that(self.flushLoggedErrors(loggedExceptionType), has_length(1))

    def test_addSuccessError(self) -> None:
        """
        If there is an error reporting success then the test run is marked as
        an error.
        """
        self._runErrorTest('test_addSuccess', AttributeError)

    def test_addErrorError(self) -> None:
        """
        If there is an error reporting an error then the test run is marked as
        an error.
        """
        self._runErrorTest('test_addError', AttributeError)

    def test_addFailureError(self) -> None:
        """
        If there is an error reporting a failure then the test run is marked
        as an error.
        """
        self._runErrorTest('test_addFailure', AttributeError)

    def test_addSkipError(self) -> None:
        """
        If there is an error reporting a skip then the test run is marked
        as an error.
        """
        self._runErrorTest('test_addSkip', AttributeError)

    def test_addExpectedFailure(self) -> None:
        """
        If there is an error reporting an expected failure then the test
        run is marked as an error.
        """
        self._runErrorTest('test_addExpectedFailure', AttributeError)

    def test_addUnexpectedSuccess(self) -> None:
        """
        If there is an error reporting an unexpected ccess then the test
        run is marked as an error.
        """
        self._runErrorTest('test_addUnexpectedSuccess', AttributeError)

    def test_failedFailureReport(self) -> None:
        """
        A failure encountered while reporting a reporting failure is logged.
        """
        worker, server, pump = connectedServerAndClient(LocalWorkerAMP, WorkerProtocol, greet=False)
        worker.transport = None
        expectedCase = pyunitcases.PyUnitTest('test_pass')
        result = TestResult()
        Deferred.fromCoroutine(server.run(expectedCase, result))
        pump.flush()
        assert_that(self.flushLoggedErrors(ConnectionLost), has_length(2))