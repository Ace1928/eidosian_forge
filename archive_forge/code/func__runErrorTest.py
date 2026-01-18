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